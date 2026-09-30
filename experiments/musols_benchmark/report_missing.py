"""Reports which cells of a sweep are missing, and emits a resubmission plan for exactly those.

A shard killed by its wall-clock limit leaves the cells it had already written and loses the rest, so a sweep
can look finished while quietly under-sampling. That matters beyond completeness: if the cells that died are
systematically the expensive ones, the surviving sample is biased toward cheap cells and the reported runtimes
understate the truth. This script names the gap so it can be closed or, at minimum, reported.

It works from the results alone plus the grid the sweep was launched with. For each shard it reports how many
of its assigned cells arrived, which lets a wall-clock kill be distinguished from a sweep that simply was not
submitted: a timed-out shard has a *prefix* of its cells, an unsubmitted one has none.

Examples:
    # What is missing from the water-reservoir sweep?
    python experiments/musols_benchmark/report_missing.py results/water-reservoir/shard_*.jsonl \\
        --env water-reservoir --seeds 0-99 --num-users 2 3 --concentrations 5 --num-shards 50

    # Same, and write a resubmission script for the shards that are short.
    python experiments/musols_benchmark/report_missing.py results/water-reservoir/shard_*.jsonl \\
        --env water-reservoir --seeds 0-99 --num-users 2 3 --concentrations 5 --num-shards 50 \\
        --emit-resubmit slurm/resubmit_water_reservoir.sh --total-timesteps 75000 --timeout 5400
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

ALGORITHMS = ("musols", "random", "vertex", "ols")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=str, nargs="+", help="Result files for the sweep (shard_*.jsonl).")
    parser.add_argument("--env", type=str, nargs="+", required=True, help="Environment key(s) the sweep ran.")
    parser.add_argument("--seeds", type=str, nargs="+", required=True, help="Seeds, each an int or 'start-end'.")
    parser.add_argument("--num-users", type=int, nargs="+", required=True)
    parser.add_argument("--concentrations", type=float, nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, default=1, help="Shard count the sweep was launched with.")
    parser.add_argument(
        "--emit-resubmit",
        type=str,
        default=None,
        help="Write a shell script that resubmits only the short shards, with --resume so finished cells are skipped.",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Passed through to the resubmission.")
    parser.add_argument("--timeout", type=float, default=None, help="Passed through to the resubmission.")
    parser.add_argument("--time-limit", type=str, default="120:00:00", help="SLURM wall-clock for the resubmission.")
    return parser.parse_args()


def parse_seed_list(values: List[str]) -> List[int]:
    seeds: List[int] = []
    for token in values:
        if "-" in token[1:]:
            start, end = token.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(token))
    return seeds


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    # The cell order must match run_study.py's, or the per-shard attribution below is meaningless.
    cells = [
        (env, seed, m, kappa)
        for env in args.env
        for seed in seeds
        for m in args.num_users
        for kappa in args.concentrations
    ]
    shard_of = {cell: i % args.num_shards for i, cell in enumerate(cells)}

    done_algos = defaultdict(set)
    broken = 0
    for pattern in args.results:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not path.exists():
                continue
            with path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        broken += 1
                        continue
                    key = (record["env"], record["seed"], record["num_users"], record["concentration"])
                    done_algos[key].add(record["algorithm"])

    expected = set(ALGORITHMS)
    complete = {c for c in cells if expected <= done_algos.get(c, set())}
    partial = {c for c in cells if c in done_algos and c not in complete}
    missing = [c for c in cells if c not in done_algos]

    print(f"Expected {len(cells)} cells across {args.num_shards} shard(s)")
    print(f"  complete : {len(complete)}")
    print(f"  partial  : {len(partial)}   (some algorithms present -- cut off mid-cell)")
    print(f"  absent   : {len(missing)}")
    if broken:
        print(f"  {broken} unparseable line(s) -- truncated writes, their cells count as incomplete")

    # Per-shard attribution: a wall-clock kill leaves a prefix, so a short shard is the signature to look for.
    per_shard = defaultdict(lambda: {"assigned": 0, "done": 0})
    for cell in cells:
        entry = per_shard[shard_of[cell]]
        entry["assigned"] += 1
        entry["done"] += cell in complete
    short = sorted(i for i, e in per_shard.items() if e["done"] < e["assigned"])

    print(f"\nShards short of their assignment: {len(short)} of {args.num_shards}")
    if short:
        print(f"  {'shard':>6}{'assigned':>10}{'done':>6}{'lost':>6}")
        for i in short:
            e = per_shard[i]
            print(f"  {i:>6}{e['assigned']:>10}{e['done']:>6}{e['assigned'] - e['done']:>6}")
        lost = sum(per_shard[i]["assigned"] - per_shard[i]["done"] for i in short)
        print(f"  total cells to recover: {lost}")

    # Whether the loss is concentrated tells you whether the surviving sample is biased.
    if missing or partial:
        by_axis = defaultdict(lambda: defaultdict(int))
        for cell in list(missing) + list(partial):
            env, seed, m, kappa = cell
            by_axis["env"][env] += 1
            by_axis["num_users"][m] += 1
            by_axis["concentration"][kappa] += 1
        print("\nWhere the gap falls (uneven distribution means the surviving sample is biased):")
        for axis, counts in by_axis.items():
            spread = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
            print(f"  {axis:<14} {spread}")

    if args.emit_resubmit:
        if not short:
            print("\nNothing to resubmit.")
            return
        flags = [f"--env {' '.join(args.env)}", f"--seeds {' '.join(args.seeds)}",
                 f"--num-users {' '.join(str(m) for m in args.num_users)}",
                 f"--concentrations {' '.join(f'{k:g}' for k in args.concentrations)}"]
        if args.total_timesteps is not None:
            flags.append(f"--total-timesteps {args.total_timesteps}")
        if args.timeout is not None:
            flags.append(f"--timeout {args.timeout:g}")
        name = args.env[0]
        script = f'''#!/usr/bin/env bash
# Resubmits ONLY the shards that fell short of their assignment, generated by report_missing.py.
#
# Every task passes --resume, so a shard re-reads its own results file and skips the cells it already
# finished: re-running is idempotent, and a shard that was killed partway continues rather than restarting.
# That is what makes it safe to resubmit a whole shard index to recover a handful of lost cells.
#
# Shards short of their assignment: {len(short)} of {args.num_shards}
# Cells to recover: {sum(per_shard[i]["assigned"] - per_shard[i]["done"] for i in short)}
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SHARDS=({" ".join(str(i) for i in short)})
ARRAY=$(IFS=,; echo "${{SHARDS[*]}}")

sbatch --job-name="musols-{name}-recover" \\
       --array="$ARRAY" \\
       --time="{args.time_limit}" \\
       study.sbatch "{name}" {args.num_shards} \\
       {" ".join(flags)} --resume

echo "Resubmitted {len(short)} shard(s): $ARRAY"
echo "When they finish:  ./collect.sh {name}"
'''
        path = Path(args.emit_resubmit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script)
        path.chmod(0o755)
        print(f"\nWrote {path} -- resubmits {len(short)} shard(s) as a sparse SLURM array.")


if __name__ == "__main__":
    main()

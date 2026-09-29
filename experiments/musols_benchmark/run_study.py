"""Runs a full randomized MUSOLS study: a sweep over seeds, panel sizes and stakeholder heterogeneity.

Where `run_experiment.py` runs one hand-specified configuration, this driver runs the experiment a paper
actually reports: for each (environment, seed, num_users m, concentration kappa) cell it samples a fresh
panel of stakeholders (see `preferences.py`), runs every algorithm on that identical panel, scores each
returned coverage set with utility-based metrics over the reachable consensus weights (see `metrics.py`), and
appends one fully self-describing record per algorithm to a JSON Lines file.

Reproducibility
----------------
Every random quantity is derived from the cell's coordinates, never from ambient state:
  - the preference matrix W comes from a generator seeded by (seed, num_users, kappa), so the same cell always
    produces the same panel, and every algorithm within a cell sees the *same* panel;
  - the evaluation weight set is seeded per cell, so all algorithms in a cell are scored on identical weights;
  - each algorithm run re-seeds the environment, the action/observation spaces and the RL solver (see
    `run_experiment._make_seeded_env`).
Each record also carries the git commit, library versions and the full hyperparameter set, so a results file
alone is enough to reconstruct exactly what produced it.

Examples:
    python experiments/musols_benchmark/run_study.py --env deep-sea-treasure --seeds 0 1 2 --num-users 2 3 \\
        --concentrations 5 50 --out results/study.jsonl
    python experiments/musols_benchmark/run_study.py --env fruit-tree --seeds 0-9 --num-users 2 4 6 --skip-ols
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import mo_gymnasium
import numpy as np
from environments import ENVIRONMENTS
from metrics import evaluate_coverage_set, exact_restricted_ccs
from preferences import preference_statistics, sample_user_weights
from run_experiment import run_experiment
from synthetic import register_synthetic_configs

from morl_baselines.common.performance_indicators import sparsity


ALGORITHMS = ("musols", "random", "vertex", "ols")


def parse_seed_list(values: List[str]) -> List[int]:
    """Expands a list of seed tokens, each either a single integer or an inclusive `start-end` range."""
    seeds: List[int] = []
    for token in values:
        if "-" in token[1:]:  # allow a leading '-' for negative ints, though seeds are normally non-negative
            start, end = token.split("-", 1) if not token.startswith("-") else token[1:].split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(token))
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, nargs="+", default=[], help="Registered environment keys to sweep.")
    parser.add_argument(
        "--synthetic-objectives",
        type=int,
        nargs="+",
        default=[],
        help="Additionally sweep the synthetic scaling task at these numbers of objectives d.",
    )
    parser.add_argument("--synthetic-candidates", type=int, default=30, help="Attainable payoffs N.")
    # "sphere" puts every candidate on the convex hull, so all N are extremal and corner-weight enumeration
    # grows combinatorially in N and d -- useful only as a deliberate worst-case stress test, never a sane
    # default. "gaussian" gives a realistic mix of extremal and dominated candidates.
    parser.add_argument("--synthetic-geometry", type=str, default="gaussian", choices=("sphere", "gaussian"))
    parser.add_argument("--seeds", type=str, nargs="+", default=["0-4"], help="Seeds; each an int or 'start-end'.")
    parser.add_argument("--num-users", type=int, nargs="+", default=[2], help="Panel sizes m to sweep.")
    parser.add_argument(
        "--concentrations",
        type=float,
        nargs="+",
        default=[5.0],
        help="Dirichlet concentrations kappa to sweep: larger means a more unanimous stakeholder panel.",
    )
    parser.add_argument("--out", type=str, required=True, help="Path of the JSON Lines results file to append to.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override the OLS/MUSOLS epsilon.")
    parser.add_argument("--timeout", type=float, default=None, help="Override the per-algorithm wall-clock budget (s).")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override the RL solver training budget.")
    parser.add_argument("--skip-ols", action="store_true", help="Skip the full-simplex OLS baseline.")
    parser.add_argument("--skip-random", action="store_true", help="Skip the Random-Omega_W baseline.")
    parser.add_argument("--skip-vertex", action="store_true", help="Skip the vertex-only baseline.")
    parser.add_argument(
        "--weight-samples",
        type=int,
        default=2000,
        help="Consensus weights drawn to estimate the utility-based metrics.",
    )
    parser.add_argument(
        "--log-trajectory",
        action="store_true",
        help="Record per-iteration quality (anytime curves). Adds one entry per evaluation per record.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard, for splitting a sweep across parallel jobs (e.g. a SLURM array task).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards the sweep is split across. Each shard runs the cells whose position "
        "in the (deterministically ordered) cell list is congruent to --shard-index modulo this.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-cell progress printing.")
    return parser.parse_args()


def _git_commit() -> Optional[str]:
    """Returns the current git commit, or None outside a repository, so results stay traceable to code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def provenance() -> dict:
    """Captures the environment a run happened in, so a results file is self-contained."""
    return {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mo_gymnasium": mo_gymnasium.__version__,
    }


def _reference_set(config, user_weights: np.ndarray, results: dict, weight_samples: int, seed: int):
    """Best available reference set for the utility-loss metric.

    For enumerable ("exact") environments this is the brute-forced ground-truth restricted CCS, so utility
    loss is measured against the true optimum. Elsewhere no ground truth exists, so the union of every
    algorithm's returned set stands in as the best-known front -- the usual convention when the true front is
    unavailable. It makes the metric comparative rather than absolute, which the record flags explicitly.
    """
    if config.solver == "exact":
        env = mo_gymnasium.make(config.env_id, **config.env_kwargs)
        candidates = np.asarray(env.unwrapped.pareto_front(gamma=config.gamma), dtype=np.float32)
        env.close()
        return exact_restricted_ccs(candidates, user_weights, num_weight_samples=weight_samples * 10, seed=seed), True
    union = [v for result in results.values() for v in result.ccs]
    return union, False


def _metric_trajectory(result, user_weights: np.ndarray, reference, args, seed: int) -> Optional[List[dict]]:
    """Converts per-iteration coverage-set snapshots into a compact anytime quality curve.

    Storing raw snapshots would bloat the results file for no benefit, so each snapshot is scored on the spot
    -- against the same reference and weight set the final metrics use -- and only the scalars are kept. The
    result is directly plottable as quality against evaluations spent.
    """
    if result.trajectory is None:
        return None
    curve = []
    for step in result.trajectory:
        step_metrics = evaluate_coverage_set(
            [np.asarray(v, dtype=np.float32) for v in step["ccs"]],
            user_weights,
            reference_set=reference,
            num_weight_samples=args.weight_samples,
            seed=seed,
        )
        curve.append(
            {
                "num_evaluated": step["num_evaluated"],
                "elapsed_seconds": step["elapsed_seconds"],
                "ccs_size": len(step["ccs"]),
                "expected_consensus_utility": step_metrics["expected_consensus_utility"],
                "max_consensus_utility_loss": step_metrics["max_consensus_utility_loss"],
            }
        )
    return curve


def run_cell(
    env_key: str,
    seed: int,
    num_users: int,
    concentration: float,
    args,
    prov: dict,
) -> List[dict]:
    """Runs every algorithm on one (environment, seed, panel size, heterogeneity) cell.

    Returns:
        One record per algorithm, ready to be written as JSON Lines.
    """
    config = ENVIRONMENTS[env_key]

    # Derived solely from the cell's coordinates: the same cell always yields the same panel, and every
    # algorithm in the cell is handed that identical panel.
    pref_rng = np.random.default_rng([seed, num_users, int(concentration * 1000)])
    user_weights = sample_user_weights(config.num_objectives, num_users, concentration, pref_rng)
    pref_stats = preference_statistics(user_weights)

    results = run_experiment(
        env_key,
        seed=seed,
        epsilon=args.epsilon,
        timeout=args.timeout,
        total_timesteps=args.total_timesteps,
        skip_ols=args.skip_ols,
        skip_random=args.skip_random,
        skip_vertex=args.skip_vertex,
        record_trajectory=args.log_trajectory,
        user_weights=user_weights,
        verbose=False,
    )

    reference, reference_is_ground_truth = _reference_set(config, user_weights, results, args.weight_samples, seed)

    records = []
    for algo in ALGORITHMS:
        if algo not in results:
            continue
        result = results[algo]
        metrics = evaluate_coverage_set(
            result.ccs,
            user_weights,
            reference_set=reference,
            num_weight_samples=args.weight_samples,
            seed=seed,
        )
        trajectory = _metric_trajectory(result, user_weights, reference, args, seed)
        records.append(
            {
                # --- cell coordinates -------------------------------------------------
                "env": env_key,
                "algorithm": algo,
                "seed": seed,
                "num_users": num_users,
                "num_objectives": config.num_objectives,
                "concentration": concentration,
                # --- the sampled panel and its realized spread ------------------------
                "user_weights": user_weights.tolist(),
                **pref_stats,
                # --- search effort ----------------------------------------------------
                "ccs_size": len(result.ccs),
                "num_evaluated": result.num_evaluated,
                "elapsed_seconds": result.elapsed_seconds,
                "converged": result.converged,
                # --- solution quality -------------------------------------------------
                **metrics,
                "sparsity": float(sparsity(result.ccs)) if len(result.ccs) > 1 else 0.0,
                "reference_is_ground_truth": reference_is_ground_truth,
                "reference_size": len(reference),
                "trajectory": trajectory,
                # --- returned solutions, so nothing needs re-running to inspect them ---
                "ccs": [np.asarray(v).tolist() for v in result.ccs],
                "weight_support": [np.asarray(w).tolist() for w in result.weight_support],
                # --- configuration and provenance -------------------------------------
                "epsilon": config.epsilon if args.epsilon is None else args.epsilon,
                "timeout_seconds": config.timeout_seconds if args.timeout is None else args.timeout,
                "solver": config.solver,
                "solver_kwargs": {
                    **config.solver_kwargs,
                    **({} if args.total_timesteps is None else {"total_timesteps": args.total_timesteps}),
                },
                "gamma": config.gamma,
                "reward_wrapper": None if config.reward_wrapper is None else config.reward_wrapper.__name__,
                "weight_samples": args.weight_samples,
                **prov,
            }
        )
    return records


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    prov = provenance()

    env_keys = list(args.env)
    if args.synthetic_objectives:
        env_keys += register_synthetic_configs(
            args.synthetic_objectives,
            num_candidates=args.synthetic_candidates,
            geometry=args.synthetic_geometry,
        )
    unknown = [key for key in env_keys if key not in ENVIRONMENTS]
    assert not unknown, f"Unknown environment key(s): {unknown}. Known: {sorted(ENVIRONMENTS)}"
    assert env_keys, "Nothing to run: pass --env and/or --synthetic-objectives."

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [(e, s, m, k) for e in env_keys for s in seeds for m in args.num_users for k in args.concentrations]
    # Sharding only selects which cells this process runs; it never changes a cell's result, because every
    # random quantity is derived from the cell's own coordinates rather than from execution order.
    assert 0 <= args.shard_index < args.num_shards, "--shard-index must be in [0, --num-shards)."
    total_cells = len(cells)
    cells = cells[args.shard_index :: args.num_shards]
    if not args.quiet:
        shard = f" (shard {args.shard_index + 1}/{args.num_shards} of {total_cells} cells)" if args.num_shards > 1 else ""
        print(f"Running {len(cells)} cells x {len(ALGORITHMS)} algorithms{shard} -> {out_path}")
        print(f"git_commit={prov['git_commit']}")

    start = time.perf_counter()
    with out_path.open("a") as handle:
        for index, (env_key, seed, num_users, concentration) in enumerate(cells, start=1):
            cell_start = time.perf_counter()
            records = run_cell(env_key, seed, num_users, concentration, args, prov)
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()  # keep partial results durable: a long sweep may be interrupted
            if not args.quiet:
                summary = "  ".join(
                    f"{r['algorithm']}: |CCS|={r['ccs_size']}, EU={r['expected_consensus_utility']:.3f}" for r in records
                )
                print(
                    f"[{index}/{len(cells)}] {env_key} seed={seed} m={num_users} kappa={concentration:g} "
                    f"({time.perf_counter() - cell_start:.1f}s)  {summary}"
                )

    if not args.quiet:
        print(f"Done in {time.perf_counter() - start:.1f}s; wrote {out_path}")


if __name__ == "__main__":
    main()

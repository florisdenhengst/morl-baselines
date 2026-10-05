#!/usr/bin/env bash
# Re-runs the ablation baselines alone, across every task, seed, kappa and m of the main study.
#
#   ./submit_ablations.sh --dry-run            # print the plan and cost, submit nothing
#   ./submit_ablations.sh                      # re-run -consensus everywhere
#   ./submit_ablations.sh water-reservoir      # one sweep only
#   ONLY="musols random" ./submit_ablations.sh # also re-run -search (see the warning below)
#
# WHY THIS EXISTS
# `VertexOnlySearch` (presented as -consensus) terminated one stakeholder early: its `ended()` tested weights
# *handed out* rather than weights *solved*, so it covered min(d,m)-1 stakeholders instead of min(d,m). Every
# -consensus row written before the fix therefore describes a weaker baseline than the paper claims. The
# signature in an existing table is an evaluation count of exactly min(d,m)-1; after the fix it reads min(d,m).
# tests/test_vertex_baseline.py pins that invariant.
#
# ONLY -consensus NEEDS RE-RUNNING. -search (RandomOmegaWSearch) was never affected, and re-running it is not
# free: its budget is MUSOLS's realized evaluation count, so it cannot run without MUSOLS, and asking for it
# drags a full MUSOLS re-run along -- on highway that is 3.7 h per evaluation. Hence the default of `vertex`
# and the opt-in ONLY override.
#
# HOW THE OLD ROWS GET REPLACED
# Nothing is deleted. Each task appends to the same shard file it wrote before, and `collect.sh` keeps the
# newest record per (algorithm, env, seed, num_users, concentration), so the fresh -consensus rows supersede
# the stale ones at merge time while every other algorithm's results stay untouched.
#
# Deliberately NOT --resume: with --only vertex, resume considers a cell done once it has a vertex row, which
# every cell already has. It would skip the entire sweep.
#
# Shard counts match submit_all.sh so each task covers the same cells as before. The cost is much lower than a
# full sweep -- one evaluation per cell instead of seven to sixteen -- so the time limits are cut accordingly.
set -euo pipefail

DRY_RUN=0
[[ ${1:-} == "--dry-run" ]] && { DRY_RUN=1; shift; }
WANTED=("$@")

ONLY=${ONLY:-vertex}

cd "$(dirname "$0")"
mkdir -p logs

SEEDS="0-99"

# name | shards | time | s/cell for ONE evaluation | run_study.py arguments (mirroring submit_all.sh)
JOBS=(
    "dst|1|04:00:00|0.1|--env deep-sea-treasure --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 30 --log-trajectory"
    "resource-gathering|4|04:00:00|2.3|--env resource-gathering --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 180 --log-trajectory"
    "fruit-tree|8|04:00:00|1|--env fruit-tree --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 120 --log-trajectory"
    "synthetic|100|08:00:00|1|--synthetic-objectives 2 3 4 5 6 7 8 --synthetic-candidates 30 --synthetic-geometry gaussian --seeds $SEEDS --num-users 2 3 --concentrations 1 5 50 --timeout 600 --log-trajectory"
    "minecart|25|24:00:00|660|--env minecart --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 7200"
    "reacher|25|24:00:00|660|--env reacher --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 7200"
    "lunar-lander|70|48:00:00|7770|--env lunar-lander --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 25200"
    "highway|50|48:00:00|13300|--env highway --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 18000"
    "water-reservoir|50|24:00:00|540|--env water-reservoir --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 75000 --timeout 5400"
)

wanted() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    for w in "${WANTED[@]}"; do [[ $w == "$1" ]] && return 0; done
    return 1
}

count_cells() {
    python - "$1" <<'PY'
import re, sys
args = sys.argv[1]
def grab(flag):
    m = re.search(rf"{flag}((?:\s+[^\s-]\S*)+)", args)
    return m.group(1).split() if m else []
n_seeds = 0
for tok in grab("--seeds"):
    if "-" in tok[1:]:
        lo, hi = tok.split("-", 1)
        n_seeds += int(hi) - int(lo) + 1
    else:
        n_seeds += 1
n_envs = max(len(grab("--env")) + len(grab("--synthetic-objectives"))
             + len(grab("--fruit-tree-depths")), 1)
print(n_envs * n_seeds * max(len(grab("--num-users")), 1) * max(len(grab("--concentrations")), 1))
PY
}

# One evaluation per retained stakeholder, so cost scales with the mean of min(d, m) over the grid. Two is a
# fair approximation across these sweeps and keeps the estimate on the pessimistic side for m=2-only grids.
EVALS_PER_CELL=2

echo "Re-running ONLY: $ONLY"
[[ $ONLY != "vertex" ]] && echo "NOTE: this includes MUSOLS or -search, which costs a full re-run of those."
echo ""
printf "%-18s %7s %10s %12s %12s %12s\n" SWEEP SHARDS CELLS "CORE-HOURS" "H/SHARD" "WALL LIMIT"
printf "%s\n" "--------------------------------------------------------------------------"
total=0; tasks=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r name shards timelimit per_eval args <<<"$job"
    wanted "$name" || continue
    cells=$(count_cells "$args")
    ch=$(python -c "print(f'{$cells * $per_eval * $EVALS_PER_CELL / 3600:.1f}')")
    per_shard=$(python -c "print(f'{$cells * $per_eval * $EVALS_PER_CELL / 3600 / $shards:.2f}')")
    printf "%-18s %7s %10s %12s %12s %12s\n" "$name" "$shards" "$cells" "$ch" "$per_shard" "$timelimit"
    total=$(python -c "print(f'{$total + $ch:.1f}')"); tasks=$((tasks + shards))

    if [[ $DRY_RUN -eq 0 ]]; then
        # shellcheck disable=SC2086 -- $args and $ONLY are intentionally word-split into separate flags
        sbatch --job-name="musols-$name-abl" \
               --array="0-$((shards - 1))" \
               --time="$timelimit" \
               study.sbatch "$name" "$shards" $args --only $ONLY
    fi
done
printf "%s\n" "--------------------------------------------------------------------------"
printf "%-18s %7s %10s %12s\n" TOTAL "$tasks" "" "$total"
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: nothing submitted. Drop --dry-run to submit."
else
    echo "Submitted. When the jobs finish:"
    echo "  ./collect.sh                     # newest rows win, so the stale -consensus rows drop out"
    echo "  python ../check_vertex_records.py 'results/*/shard_*.jsonl'   # confirm by commit/date"
    echo "Then verify the evals column reads min(d,m), not min(d,m)-1."
fi

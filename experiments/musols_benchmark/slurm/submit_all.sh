#!/usr/bin/env bash
# Submits the full paper-scale MUSOLS study as SLURM array jobs.
#
#   ./submit_all.sh --dry-run     # print the plan and the cost estimate, submit nothing
#   ./submit_all.sh               # submit everything
#   ./submit_all.sh synthetic     # submit only the named sweep(s)
#
# Every sweep uses n=100 sampled stakeholder panels per configuration. Shard counts and time limits below are
# derived from per-cell runtimes measured on a single workstation core (see README.md for the measurements and
# the arithmetic); they include roughly a 2x safety factor, because a sweep that overruns its wall-clock limit
# loses only the shards still running, and those can be resubmitted individually.
set -euo pipefail

DRY_RUN=0
[[ ${1:-} == "--dry-run" ]] && { DRY_RUN=1; shift; }
WANTED=("$@")

cd "$(dirname "$0")"
mkdir -p logs

SEEDS="0-99"   # n=100 sampled panels per configuration

# name | shards | time | measured s/cell | run_study.py arguments
#
# Grid choice differs by cost, deliberately. The cheap environments get the full m x kappa sweep, because it
# is nearly free and the heterogeneity axis is one of the paper's contributions. The expensive deep-RL
# environments get m in {2,3} at a single kappa: n=100 panels per configuration is what statistical power
# needs, and spending the remaining budget on more kappa values there would cost more than it reveals.
JOBS=(
    # --- exact / synthetic: cheap, so sweep everything, and record anytime trajectories --------------------
    "dst|1|00:30:00|0.3|--env deep-sea-treasure --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --log-trajectory"
    "resource-gathering|4|01:00:00|5.6|--env resource-gathering --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --log-trajectory"
    "fruit-tree|8|02:00:00|21|--env fruit-tree --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --log-trajectory"

    # --- the scalability grid: m in {2,3} as requested, d well past any public benchmark ------------------
    # OLS is given a 600 s budget per algorithm. It will exhaust it from about d=6 onward; that is the result,
    # not a failure, and `converged` records it. Ratios from censored cells are lower bounds -- analyze.py
    # reports the converged fraction alongside them so the two are never confused.
    "synthetic|100|08:00:00|265|--synthetic-objectives 2 3 4 5 6 7 8 --synthetic-candidates 30 --synthetic-geometry gaussian --seeds $SEEDS --num-users 2 3 --concentrations 1 5 50 --timeout 600 --log-trajectory"

    # --- deep RL: budgets raised to convergence-plausible values (see README.md) --------------------------
    "minecart|16|04:00:00|440|--env minecart --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 100000 --timeout 3600"
    "reacher|16|04:00:00|440|--env reacher --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 100000 --timeout 3600"
    "lunar-lander|60|08:00:00|4275|--env lunar-lander --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 100000 --timeout 3600"
    "highway|50|08:00:00|2430|--env highway --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 50000 --timeout 3600"
    "water-reservoir|50|08:00:00|2384|--env water-reservoir --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 75000 --timeout 3600"
)

wanted() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    for w in "${WANTED[@]}"; do [[ $w == "$1" ]] && return 0; done
    return 1
}

# Counts the cells a sweep expands to, so the cost estimate reflects the actual grid rather than a guess.
count_cells() {
    local args=$1
    python - "$args" <<'PY'
import re, sys
args = sys.argv[1]
def grab(flag):
    m = re.search(rf"{flag}((?:\s+[^\s-]\S*)+)", args)
    return m.group(1).split() if m else []
seeds = grab("--seeds")
n_seeds = 0
for tok in seeds:
    if "-" in tok[1:]:
        lo, hi = tok.split("-", 1)
        n_seeds += int(hi) - int(lo) + 1
    else:
        n_seeds += 1
n_envs = max(len(grab("--env")) + len(grab("--synthetic-objectives")), 1)
n_users = max(len(grab("--num-users")), 1)
n_kappa = max(len(grab("--concentrations")), 1)
print(n_envs * n_seeds * n_users * n_kappa)
PY
}

total_core_hours=0
total_tasks=0
printf "%-18s %7s %10s %12s %14s %12s\n" SWEEP SHARDS CELLS "CORE-HOURS" "H/SHARD" "WALL LIMIT"
printf "%s\n" "----------------------------------------------------------------------------------"

for job in "${JOBS[@]}"; do
    IFS='|' read -r name shards timelimit per_cell args <<<"$job"
    wanted "$name" || continue
    cells=$(count_cells "$args")
    core_hours=$(python -c "print(f'{$cells * $per_cell / 3600:.1f}')")
    per_shard=$(python -c "print(f'{$cells * $per_cell / 3600 / $shards:.2f}')")
    printf "%-18s %7s %10s %12s %14s %12s\n" "$name" "$shards" "$cells" "$core_hours" "$per_shard" "$timelimit"
    total_core_hours=$(python -c "print(f'{$total_core_hours + $core_hours:.1f}')")
    total_tasks=$((total_tasks + shards))

    if [[ $DRY_RUN -eq 0 ]]; then
        # shellcheck disable=SC2086 -- $args is intentionally word-split into separate flags
        sbatch --job-name="musols-$name" \
               --array="0-$((shards - 1))" \
               --time="$timelimit" \
               study.sbatch "$name" "$shards" $args
    fi
done

printf "%s\n" "----------------------------------------------------------------------------------"
printf "%-18s %7s %10s %12s\n" TOTAL "$total_tasks" "" "$total_core_hours"
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: nothing submitted. Drop --dry-run to submit."
else
    echo "Submitted. Merge shards when the jobs finish:  ./collect.sh"
fi

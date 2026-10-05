#!/usr/bin/env bash
# Submits the full paper-scale MUSOLS study as SLURM array jobs.
#
#   ./submit_all.sh --dry-run     # print the plan and the cost estimate, submit nothing
#   ./submit_all.sh               # submit everything
#   ./submit_all.sh synthetic     # submit only the named sweep(s)
#
# This file covers the sweeps the paper *reports*. The hopper continuous-control demo is deliberately not
# here: it is illustrative rather than evaluative and is submitted separately via ./submit_hopper_demo.sh.
#
# Every sweep uses n=100 sampled stakeholder panels per configuration. Each job passes an explicit
# --timeout: a per-algorithm wall-clock cap within one cell, sized from the solver and the costs measured for
# that environment (see README.md "Where the budgets come from"). It is set explicitly here rather than left
# to fall back to environments.py's own EnvironmentConfig.timeout_seconds defaults, which are inconsistent
# across environments and, for fruit-tree, were too tight to let OLS ever converge (see the fruit-tree line
# below). Every shard's SLURM wall-clock is separately set to 120:00:00 -- the cluster's maximum -- as a
# backstop: even in the pessimistic case where every algorithm in every cell exhausts its --timeout without
# converging, no shard in this file comes close to that ceiling (worst case is lunar-lander's, at ~27h/shard).
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
    # --timeout values below are per-algorithm caps within a cell, explicit here rather than left to fall
    # back to environments.py's own (inconsistent) EnvironmentConfig.timeout_seconds defaults -- see README.md
    # "Where the budgets come from" for the measurements each one is based on.
    "dst|1|120:00:00|0.3|--env deep-sea-treasure --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 30 --log-trajectory"
    "resource-gathering|4|120:00:00|5.6|--env resource-gathering --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 180 --log-trajectory"
    "fruit-tree|8|120:00:00|21|--env fruit-tree --seeds $SEEDS --num-users 2 3 4 --concentrations 1 5 50 --timeout 120 --log-trajectory"

    # --- the scalability grid: m in {2,3} as requested, d well past any public benchmark ------------------
    # OLS is given a 600 s budget per algorithm. It will exhaust it from about d=6 onward; that is the result,
    # not a failure, and `converged` records it. Ratios from censored cells are lower bounds -- analyze.py
    # reports the converged fraction alongside them so the two are never confused.
    "synthetic|100|120:00:00|265|--synthetic-objectives 2 3 4 5 6 7 8 --synthetic-candidates 30 --synthetic-geometry gaussian --seeds $SEEDS --num-users 2 3 --concentrations 1 5 50 --timeout 600 --log-trajectory"

    # --- deep RL: 500k-step training budgets (see README.md) ---------------------------------------------
    # s/cell below is the old measurement scaled by the budget increase; --timeout is sized so that even the
    # pessimistic case -- every one of the 4 algorithms in every cell of a shard exhausting its cap without
    # converging -- stays under the 120 h shard wall-clock. Shard counts were raised where that margin got
    # tight: worst case is now 64 h/shard (minecart, reacher), 80 h (lunar-lander, highway), 96 h (water).
    "minecart|25|120:00:00|2200|--env minecart --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 7200"
    "reacher|25|120:00:00|4400|--env reacher --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 7200"
    "lunar-lander|70|120:00:00|21375|--env lunar-lander --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 25200"
    "highway|50|120:00:00|24300|--env highway --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 18000"
    # Raised 75k -> 500k to match the other deep-RL rows, with eval_episodes=50 set in environments.py. The
    # per-algorithm cap had to go up with it: MUSOLS ran 3.53 evaluations in a median 1592 s at 75k, i.e.
    # ~450 s per evaluation, so 500k puts an evaluation near 3000 s and the old 5400 s cap would have
    # censored MUSOLS itself -- which is the one thing that must not happen, since MUSOLS converging on
    # 96.6% of cells is what makes its numbers readable. 21600 s leaves it ~7 evaluations against the ~3.5
    # it needs. OLS stays censored at the cap, as it already was at 75k (0/177 converged); that is accepted
    # and `converged` records it. Expected ~48600 s/cell, so 4 cells/shard is ~54 h, worst case 96 h.
    "water-reservoir|50|120:00:00|48600|--env water-reservoir --seeds $SEEDS --num-users 2 3 --concentrations 5 --total-timesteps 500000 --timeout 21600"
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

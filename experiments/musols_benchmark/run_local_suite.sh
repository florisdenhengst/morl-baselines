#!/usr/bin/env bash
# Runs the experiments that fit on a single workstation, cheapest and highest-value first, so that an
# interrupted run still leaves the most important results on disk. Each stage appends to its own JSON Lines
# file and run_study.py flushes after every cell, so partial output is always usable.
#
# The expensive RL environments (water-reservoir, minecart, highway, reacher) are included only at reduced
# training budgets: at their configured budgets a single cell costs tens of minutes to hours, which belongs on
# a cluster. Those stages validate the pipeline and produce indicative numbers, not publishable ones.
set -u

PY=/Users/floris/miniconda3/envs/morl_bl_musols/bin/python
STUDY="$(dirname "$0")/run_study.py"
OUT=${1:-results}
mkdir -p "$OUT"

stage() {
    local name=$1; shift
    echo ""
    echo "=================================================================="
    echo "STAGE: $name   ($(date +%H:%M:%S))"
    echo "=================================================================="
    "$PY" "$STUDY" "$@" 2>&1 | grep -viE "userwarning|gym\.logger|deprecat|^ *warnings\.warn|^removed value"
    echo "STAGE $name finished at $(date +%H:%M:%S)"
}

# ---------------------------------------------------------------- exact solvers: full seeds, full sweeps
stage "exact-cheap (DST + resource-gathering)" \
    --env deep-sea-treasure resource-gathering \
    --seeds 0-9 --num-users 2 3 4 --concentrations 1 5 50 \
    --log-trajectory --out "$OUT/exact_cheap.jsonl"

stage "fruit-tree (d=6, OLS hits its timeout every cell)" \
    --env fruit-tree \
    --seeds 0-9 --num-users 2 3 4 --concentrations 1 5 50 \
    --log-trajectory --out "$OUT/fruit_tree.jsonl"

# ---------------------------------------------------------------- the scalability grid (headline figure)
stage "synthetic scalability (d x m)" \
    --synthetic-objectives 2 3 4 --synthetic-candidates 20 \
    --seeds 0-4 --num-users 2 3 5 --concentrations 1 5 50 \
    --log-trajectory --out "$OUT/scaling.jsonl"

# ---------------------------------------------------------------- deep RL: affordable one first
stage "lunar-lander (SAC, configured budget)" \
    --env lunar-lander \
    --seeds 0 1 2 --num-users 2 3 --concentrations 5 \
    --out "$OUT/lunar_lander.jsonl"

# ---------------------------------------------------------------- new envs, reduced budget (indicative only)
stage "new envs at reduced budget (indicative)" \
    --env minecart reacher \
    --seeds 0 1 --num-users 2 3 --concentrations 5 \
    --total-timesteps 5000 --timeout 900 \
    --out "$OUT/new_envs_reduced.jsonl"

stage "highway at reduced budget (indicative)" \
    --env highway \
    --seeds 0 --num-users 2 3 --concentrations 5 \
    --total-timesteps 5000 --timeout 900 \
    --out "$OUT/highway_reduced.jsonl"

# ---------------------------------------------------------------- most expensive, last on purpose
stage "water-reservoir (SAC, configured budget, single cell)" \
    --env water-reservoir \
    --seeds 0 --num-users 2 --concentrations 5 \
    --out "$OUT/water_reservoir.jsonl"

echo ""
echo "ALL STAGES COMPLETE at $(date +%H:%M:%S)"
wc -l "$OUT"/*.jsonl

#!/bin/bash
# Trains the showcase demo: one realistic stakeholder panel, MUSOLS vs OLS, policies kept.
#
#   sbatch submit_demo.sbatch                       # hopper, the default
#   sbatch submit_demo.sbatch fruit-tree demo_ft    # any other registered environment
#   TIMEOUT=43200 sbatch submit_demo.sbatch         # override the per-algorithm wall-clock cap
#   SOLVER_KWARGS='{"num_seeds":2}' sbatch submit_demo.sbatch   # override inner-loop hyperparameters
#   RECORD_VIDEO=1 sbatch submit_demo.sbatch        # also render a rollout per policy (needs working OSMesa)
#
# This is not part of the evaluation and is deliberately not in submit_all.sh: it produces the artifact behind
# a talk slide or the paper's webpage, not a reported number. It is a single task rather than an array,
# because a demo is one panel -- there is nothing to shard.
#
# Cost on hopper, measured on a Genoa core by job 27373892: **18780 s per 1M-step evaluation**, 5.1x the
# 3658 s that README.md's table predicts -- and num_seeds below multiplies that. At num_seeds=2 a MUSOLS
# evaluation costs ~10.4 h, so its ~3 weights take ~31 h, and OLS spends whatever TIMEOUT allows. The wall
# clock is the cluster maximum rather than the 24 h first assumed.
#
# That first attempt returned nothing usable -- |CCS|=1 for both algorithms, `converged=false`, one
# evaluation each -- because this script passed no --timeout and so inherited hopper's
# `EnvironmentConfig.timeout_seconds` of 1800 s from environments.py. run_outer_loop checks the clock
# *before* each solve, so a single 18780 s evaluation ran and the loop then exited immediately, every time.
# Every sweep in submit_all.sh passes an explicit --timeout for exactly this reason; this script did not.
#
# Output lands in $MUSOLS_REPO/demo/<name>/: demo.json plus one checkpoint per coverage-set policy. demo.json
# carries W and both coverage sets, which is all a page needs to resolve a consensus slider client-side --
# for alpha in the simplex the chosen policy is argmax_v (W @ alpha) . v, no inference required.

#SBATCH --job-name=musols-demo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --partition=genoa
# #SBATCH --account=your-account

set -euo pipefail

ENV_KEY=${1:-hopper}
OUT_NAME=${2:-demo_$ENV_KEY}

# Per-algorithm wall-clock cap. Sized for hopper at 18780 s/evaluation: 24 h lets MUSOLS converge (~3-4
# evaluations) while capping OLS, whose search over the full simplex will not converge at this budget. That
# asymmetry is the demo's point rather than a problem -- but it does make OLS's numbers *censored*, so its
# wasted-policy count is a lower bound and `speedup` is meaningless when both algorithms hit the cap. Check
# `converged` in demo.json before quoting either.
TIMEOUT=${TIMEOUT:-86400}

# Inner-loop hyperparameter overrides, merged over the environment's registry defaults. Two matter here and
# both are defaulted up from the study's settings, because a demo is a handful of solver calls whose failure
# is unrecoverable rather than 100 panels whose noise averages out:
#
#   num_seeds=2     -- take the best of 2 independent trainings per weight. OLS/MUSOLS assume an *exact*
#                      inner solver; a training run that lands badly closes a corner weight that should have
#                      stayed open, and nothing can reopen it. MUSOLS is the more exposed, since making few
#                      solver calls is the entire point and leaves it no redundancy. Costs 2x per evaluation.
#   eval_episodes=30 -- the default of 5 episodes is a thin Monte Carlo estimate of a stochastic policy's
#                      return, and those estimates are the geometry OLS computes corner weights from.
#                      Rollouts are a rounding error against a 5 h training run.
#
# Both default to 1 / 5 in solvers.py so the study's existing results stay comparable; they are raised here
# rather than in environments.py for the same reason.
SOLVER_KWARGS=${SOLVER_KWARGS:-'{"num_seeds":2,"eval_episodes":30}'}

# Unlike study.sbatch this is a single job, not one of hundreds of array tasks sharing a node, so there is no
# throughput collapse to protect against and no reason to pin torch to one thread. SAC on a [256,256] MLP
# parallelizes only modestly, so do not expect this to scale with the core count.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
# See study.sbatch: MuJoCo picks its GL backend when the module is imported, so "osmesa" breaks `import
# mujoco` on a node lacking the OSMesa library even for a run that renders nothing. Training needs no GL.
# Pass RECORD_VIDEO=1 to request gait rollouts -- that does need a working backend, so it is opt-in and
# failures there are contained (run_demo.py skips a video rather than losing the numeric artifact).
if [[ ${RECORD_VIDEO:-0} == 1 ]]; then
    export MUJOCO_GL=${MUJOCO_GL:-osmesa}
    VIDEO_FLAG="--record-video"
else
    export MUJOCO_GL=disable
    VIDEO_FLAG=""
fi
export SDL_VIDEODRIVER=dummy

# ---------------------------------------------------------------- adapt this block to your cluster
module load 2025
module load CUDA/12.8.0

MUSOLS_CONDA=${MUSOLS_CONDA:-$HOME/miniforge3}
MUSOLS_ENV=${MUSOLS_ENV:-morl_bl_musols}
source "$MUSOLS_CONDA/etc/profile.d/conda.sh"
conda activate "$MUSOLS_ENV"
# -------------------------------------------------------------------------------------------------

REPO_ROOT=${MUSOLS_REPO:-/home/fdenhengst/MUSOLS/morl-baselines}
OUT_DIR="$REPO_ROOT/demo/$OUT_NAME"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} env=$ENV_KEY out=$OUT_DIR"

# shellcheck disable=SC2086 -- VIDEO_FLAG is empty or a single flag
python "$REPO_ROOT/experiments/musols_benchmark/run_demo.py" \
    --env "$ENV_KEY" \
    --timeout "$TIMEOUT" \
    --solver-kwargs "$SOLVER_KWARGS" \
    --out "$OUT_DIR" $VIDEO_FLAG

echo ""
echo "Demo artifact written. Fetch it with:"
echo "  scp -r <cluster>:$OUT_DIR ./demo/"
echo ""
echo "Then render the gait clips locally, where a GL backend is a given:"
echo "  python experiments/musols_benchmark/render_demo_videos.py demo/$OUT_NAME --deterministic"

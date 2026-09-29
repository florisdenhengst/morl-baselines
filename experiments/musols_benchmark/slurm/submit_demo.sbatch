#!/bin/bash
# Trains the showcase demo: one realistic stakeholder panel, MUSOLS vs OLS, policies kept.
#
#   sbatch submit_demo.sbatch                       # hopper, the default
#   sbatch submit_demo.sbatch fruit-tree demo_ft    # any other registered environment
#
# This is not part of the evaluation and is deliberately not in submit_all.sh: it produces the artifact behind
# a talk slide or the paper's webpage, not a reported number. It is a single task rather than an array,
# because a demo is one panel -- there is nothing to shard.
#
# Cost on hopper: MUSOLS needs ~3 policy trainings and OLS ~8, at a measured 3658 s per 1M-step evaluation,
# so roughly 11 evaluations ~= 11 h in total. The 24 h limit leaves room for OLS to need more than expected.
#
# Output lands in $MUSOLS_REPO/demo/<name>/: demo.json plus one checkpoint per coverage-set policy. demo.json
# carries W and both coverage sets, which is all a page needs to resolve a consensus slider client-side --
# for alpha in the simplex the chosen policy is argmax_v (W @ alpha) . v, no inference required.

#SBATCH --job-name=musols-demo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=genoa
# #SBATCH --account=your-account

set -euo pipefail

ENV_KEY=${1:-hopper}
OUT_NAME=${2:-demo_$ENV_KEY}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=osmesa
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

python "$REPO_ROOT/experiments/musols_benchmark/run_demo.py" \
    --env "$ENV_KEY" \
    --out "$OUT_DIR"

echo ""
echo "Demo artifact written. Fetch it with:"
echo "  scp -r <cluster>:$OUT_DIR ./demo/"

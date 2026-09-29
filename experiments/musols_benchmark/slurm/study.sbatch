#!/bin/bash
# Generic SLURM array job for one MUSOLS study sweep.
#
# Each array task runs one shard of the sweep's cells and writes its own results file. Sharding never changes
# a cell's result -- every random quantity is derived from the cell's own coordinates -- so the union of the
# shards is exactly what a single serial run would have produced.
#
# Usage (normally via submit_all.sh):
#   sbatch --array=0-9 --time=04:00:00 study.sbatch <name> <num_shards> <run_study.py args...>
#
# Cluster-specific settings (partition, account, the environment activation below) are intentionally left for
# you to fill in: they cannot be guessed. Everything else is portable.

#SBATCH --job-name=musols
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# --array and --time are supplied at submission time, since they differ per sweep.
# Add your scheduler's requirements here, e.g.:
# #SBATCH --partition=compute
# #SBATCH --account=your-account

set -euo pipefail

NAME=$1
NUM_SHARDS=$2
shift 2
STUDY_ARGS=("$@")

# One core per task is deliberate. These runs are small-network SAC and polytope vertex enumeration, neither
# of which parallelizes well, and torch otherwise grabs every core on the node -- which collapses throughput
# when hundreds of array tasks share a machine. Many single-core tasks beat few multi-core ones here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=osmesa   # MuJoCo must not try to open a display on a headless node.
export SDL_VIDEODRIVER=dummy  # highway-env pulls in pygame, which does the same.

# ---------------------------------------------------------------- adapt this block to your cluster
# module load python/3.12
source "${MUSOLS_CONDA:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${MUSOLS_ENV:-morl_bl_musols}"
# -------------------------------------------------------------------------------------------------

REPO_ROOT=${MUSOLS_REPO:-$PWD}
RESULT_DIR="$REPO_ROOT/results/$NAME"
mkdir -p "$RESULT_DIR"

SHARD=${SLURM_ARRAY_TASK_ID:-0}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} shard=$SHARD/$NUM_SHARDS name=$NAME"
echo "args: ${STUDY_ARGS[*]}"

srun python "$REPO_ROOT/experiments/musols_benchmark/run_study.py" \
    "${STUDY_ARGS[@]}" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --out "$RESULT_DIR/shard_$(printf '%04d' "$SHARD").jsonl"

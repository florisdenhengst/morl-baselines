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
#SBATCH --partition=genoa
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
# MuJoCo resolves a GL backend at *import* time, not when something is rendered. "osmesa" therefore makes
# `import mujoco` fail outright on a node without the OSMesa shared library (PyOpenGL raises
# "'NoneType' object has no attribute 'glGetError'"), even though these runs never render a frame. "disable"
# skips GL initialization entirely, which is what a training-only job wants. Set it to osmesa/egl only for a
# job that actually renders, and only where that backend is known to work.
export MUJOCO_GL=disable
export SDL_VIDEODRIVER=dummy  # highway-env pulls in pygame, which does the same.

# ---------------------------------------------------------------- adapt this block to your cluster
module load 2025
module load CUDA/12.8.0

# MUSOLS_CONDA is the conda *installation prefix*; MUSOLS_ENV is the environment name inside it.
MUSOLS_CONDA=${MUSOLS_CONDA:-$HOME/miniforge3}
MUSOLS_ENV=${MUSOLS_ENV:-morl_bl_musols}
source "$MUSOLS_CONDA/etc/profile.d/conda.sh"
conda activate "$MUSOLS_ENV"
# -------------------------------------------------------------------------------------------------

# Repository root -- the directory containing experiments/ and morl_baselines/, not the slurm/ subdirectory.
REPO_ROOT=${MUSOLS_REPO:-/home/fdenhengst/MUSOLS/morl-baselines}
RESULT_DIR="$REPO_ROOT/results/$NAME"
mkdir -p "$RESULT_DIR"

SHARD=${SLURM_ARRAY_TASK_ID:-0}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} shard=$SHARD/$NUM_SHARDS name=$NAME"
echo "args: ${STUDY_ARGS[*]}"

# Run python directly rather than under srun. With one task and one core, srun launches nothing it needs to
# and creates a job *step*: failures are then recorded against step .0 and terminate the step, which is how a
# single out-of-memory task ends up reported as a step failure with the batch script's own exit path obscured.
# Without it, the process is the batch step, its exit code is the job's, and accounting (MaxRSS) lands on
# .batch where it is easier to read. CPU affinity still comes from the cgroup that --cpus-per-task sets.
python "$REPO_ROOT/experiments/musols_benchmark/run_study.py" \
    "${STUDY_ARGS[@]}" \
    --shard-index "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --out "$RESULT_DIR/shard_$(printf '%04d' "$SHARD").jsonl"

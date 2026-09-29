# Running the MUSOLS study on SLURM

Full paper-scale sweep: **n=100 sampled stakeholder panels per configuration**, every environment, every
baseline. Roughly **3350 core-hours across 333 array tasks**. The deep-RL sweeps dominate that figure: at
500k-step training budgets, lunar-lander and highway alone account for about 2540 of those core-hours.

```bash
cd experiments/musols_benchmark/slurm
./submit_all.sh --dry-run      # print the plan and cost estimate, submit nothing
./submit_all.sh                # submit everything
./submit_all.sh synthetic      # submit one sweep
./collect.sh --check           # completeness report once jobs finish
./collect.sh                   # merge shards into results/merged/
```

Then regenerate every table and figure:

```bash
python experiments/musols_benchmark/analyze.py results/merged/*.jsonl
python experiments/musols_benchmark/make_paper_assets.py results/merged/*.jsonl --out paper_assets
```

## Adapt these before submitting

`study.sbatch` is currently set up for Snellius: it runs `module load 2025` and `module load cuda`, requests
the `genoa` partition, and defaults `MUSOLS_REPO` to `/home/fdenhengst/MUSOLS/morl-baselines`. On any other
cluster, three things must be changed:

1. **`study.sbatch`** — your `--partition` / `--account`, and the `module load` lines.
2. **`MUSOLS_REPO`** — the repository root, i.e. the directory holding `experiments/` and `morl_baselines/`
   (*not* the `slurm/` subdirectory: the script appends `experiments/musols_benchmark/run_study.py` to it).
3. **`MUSOLS_CONDA` / `MUSOLS_ENV`** — conda *installation prefix* and *environment name*, two different
   things (default `~/miniconda3` and `morl_bl_musols`).

`module load cuda` is harmless but not needed: every task is submitted without a GPU, so torch runs on CPU
regardless. Drop it if your scheduler charges for the module set.

The job requests **1 core and 8 GB per task, no GPU**, and pins `OMP_NUM_THREADS=1`. That pinning is not
incidental: these runs are small-network SAC plus polytope vertex enumeration, neither of which parallelizes
well, and torch will otherwise seize every core on the node and collapse throughput once hundreds of array
tasks share machines. `MUJOCO_GL=osmesa` and `SDL_VIDEODRIVER=dummy` keep MuJoCo and highway-env's pygame
dependency from trying to open a display on a headless node.

## Where the budgets come from

Per-cell costs (one cell = all four algorithms on one panel) were measured on a single workstation core:

| Sweep | measured s/cell | basis |
|---|---|---|
| deep-sea-treasure | 0.3 | 90 cells in the local suite |
| resource-gathering | 5.6 | 90 cells, tabular Q at 30k steps |
| fruit-tree | 21 | 90 cells, but with a since-fixed 20 s per-algorithm cap that was too tight for OLS |
| synthetic | 0.5 / 4.5 / 40 at d=2/3/4 | measured; ×~9 per added dimension |
| minecart | 2200 | 22 s at 5k steps, scaled to 500k |
| reacher | 4400 | 22 s at 5k steps, scaled to 500k |
| lunar-lander | 21375 | 342 s at 8k steps, scaled to 500k |
| highway | 24300 | 243 s at 5k steps, scaled to 500k |
| water-reservoir | 2384 | one full cell at 75k steps |

These are linear extrapolations and run pessimistic: minecart's real per-evaluation cost came in at ~130–190 s
against the 440 s its row predicted at 100k, so treat the deep-RL rows as an upper bound on the true spend.

Every job passes an explicit `--timeout`: a per-algorithm wall-clock cap within one cell, sized above from the
solver and the measured costs in the table above (with roughly a 30–60× margin for the cheap/exact
environments and a smaller but still comfortable margin for the deep-RL ones). It is set explicitly in
`submit_all.sh` rather than left to fall back to each environment's own `EnvironmentConfig.timeout_seconds` in
`environments.py`, which are inconsistent across environments and, for fruit-tree, were too tight for OLS to
ever converge (its 20 s default is almost exactly fruit-tree's observed OLS runtime, and `converged` was
`0/900`). Every shard's SLURM wall-clock is separately set to `120:00:00` — the cluster's maximum — as a
backstop, and the deep-RL caps and shard counts are sized together so that even the fully pessimistic case,
where all four algorithms in every cell of a shard exhaust the cap without converging, stays under it: 64 h
for minecart and reacher, 80 h for lunar-lander and highway, 24 h for water-reservoir. (Shard counts were
raised from 16→25 for minecart/reacher and 60→70 for lunar-lander to keep that margin once budgets went to
500k.) A shard that is nonetheless killed can be resubmitted individually --
sharding never changes a cell's result, only which process computes it -- but check `sacct`/the shard's
`.out` log for `TIMEOUT` before assuming a short results file just means fewer cells were assigned to it.

### Training budgets were raised for the paper runs

Every deep-RL environment except water-reservoir now trains for **500k steps**, with per-environment SAC
settings chosen for how each one actually fails rather than one global default:

- **minecart (100k → 500k)**, `learning_starts=10_000`, `buffer_size=500_000`, γ=0.98. Fuel is charged every
  step but ore pays only on returning to base, so without a long pure-random warmup SAC collapses onto doing
  nothing before it has ever seen a completed mine-and-return. The large buffer keeps those rare early
  successes sampleable late in the run.
- **lunar-lander (100k → 500k)**, `learning_starts=10_000`, `buffer_size=500_000`, γ=0.99 (its 1000-step
  episode limit needs the long horizon). Its reward normalization was also corrected — the old fuel divisor
  of 25 inflated fuel to ~1.9x the magnitude of landing success, which rewards crashing promptly over firing
  the engines; all four objectives now share a divisor of 100.
- **reacher (50k → 500k)**, `buffer_size=100_000`, γ=0.98. Episodes reset after ~50 steps, so a 0.99 horizon
  reaches well past the episode; the state-action space is small enough that 100k transitions is ample.
- **highway (50k → 500k)**, `learning_starts=5_000`, `buffer_size=500_000`, **`autotune=False, alpha=0.05`**,
  plus 4-frame observation stacking and a new reward scaler. Three separate fixes, all aimed at the same
  failure: collisions end episodes instantly, so aggressive early exploration fills the buffer with crash
  stubs. MOSACDiscrete's autotuned entropy coefficient *starts at exp(0)=1.0* and ignores the `alpha` argument
  while autotune is on, so disabling autotune is the only way to actually start conservative. Frame stacking
  makes relative vehicle velocities observable from a single input instead of inferable from one snapshot.
- **water-reservoir stays at 75k**, where we verified policies genuinely differentiate by weight vector.

Minecart is sparse-reward and even 500k may be low; if its policies still look untrained (a `|CCS|` of 1 for
every algorithm is the tell), that budget is the first thing to raise again.

### Grid differs by cost, on purpose

Cheap environments get the full `m ∈ {2,3,4} × κ ∈ {1,5,50}` sweep (900 cells) plus anytime trajectories. The
expensive deep-RL environments get `m ∈ {2,3}` at a single κ (200 cells) and no trajectories: n=100 panels per
configuration is what statistical power requires, and extra κ values there would cost more than they reveal.

## The scalability sweep

`d ∈ {2,…,8}` at `m ∈ {2,3}`, which takes the sweep well past d=6 — the ceiling of every public MORL
benchmark. MUSOLS's cost is roughly flat in d (it searches a 1–2 dimensional consensus space), while OLS grows
about 9× per added dimension.

**OLS is given a 600 s budget per algorithm and will exhaust it from about d=6 onward.** That is the finding
rather than a failure, and `converged` records it per cell — but it means cost ratios from those cells are
**lower bounds**, since OLS never finished. Report them as censored, and read the converged fraction
alongside: it falls to zero exactly where the unrestricted search stops being viable. (600 s was chosen from a
direct probe at d=8: after a full 600 s, OLS's pending-corner-weight queue was still *growing*, not shrinking
-- more time would not have materially closed the gap, so the cap is not the binding limitation here, the
combinatorics are.)

## Reproducibility

Sharding selects *which* cells a process runs and never changes a cell's result — every random quantity is
derived from the cell's own coordinates, not from execution order. The union of shards is therefore exactly
what one serial run would have produced, and a failed shard can be resubmitted alone. Each record carries the
git commit, library versions and full hyperparameters, so a results file is self-describing.

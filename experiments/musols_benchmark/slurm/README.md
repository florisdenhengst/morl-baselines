# Running the MUSOLS study on SLURM

Full paper-scale sweep: **n=100 sampled stakeholder panels per configuration**, every environment, every
baseline. Roughly **870 core-hours across 305 array tasks** — about a day of wall-clock on a modest
allocation.

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

Three things cannot be guessed and must be set for your cluster:

1. **`study.sbatch`** — add your `--partition` / `--account`, and fix the environment activation block.
2. **`MUSOLS_REPO`** — export the repository root if you submit from elsewhere (defaults to `$PWD`).
3. **`MUSOLS_CONDA` / `MUSOLS_ENV`** — conda prefix and environment name (default `~/miniconda3`, `morl_bl_musols`).

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
| fruit-tree | 21 | 90 cells; OLS exhausts its 20 s budget every cell |
| synthetic | 0.5 / 4.5 / 40 at d=2/3/4 | measured; ×~9 per added dimension |
| minecart | 440 | 22 s at 5k steps, scaled to 100k |
| reacher | 440 | 22 s at 5k steps, scaled to 100k |
| lunar-lander | 4275 | 342 s at 8k steps, scaled to 100k |
| highway | 2430 | 243 s at 5k steps, scaled to 50k |
| water-reservoir | 2384 | one full cell at 75k steps |

Shard counts target ~2–4 h per shard, and wall-clock limits carry roughly a 2× safety factor. Overrunning
costs only the shards still running, and those resubmit individually.

### Training budgets were raised for the paper runs

The budgets used during development were smoke-test values, too low to support a claim. For these runs:

- **lunar-lander 8k → 100k.** SAC does not come close to solving lunar lander at 8k; this is the single
  largest cost increase in the sweep (237 core-hours) and the one most worth it.
- **reacher 50k → 100k.** Reacher's simulation is essentially free, so the cost is network updates only.
- **minecart stays at 100k**, **water-reservoir at 75k** (where we verified policies actually differentiate
  by weight vector), **highway at 50k** (its 8.3 ms/step simulation dominates).

Minecart is sparse-reward and 100k may still be low; if its policies look untrained, that budget is the first
thing to raise.

### Grid differs by cost, on purpose

Cheap environments get the full `m ∈ {2,3,4} × κ ∈ {1,5,50}` sweep (900 cells) plus anytime trajectories. The
expensive deep-RL environments get `m ∈ {2,3}` at a single κ (200 cells) and no trajectories: n=100 panels per
configuration is what statistical power requires, and extra κ values there would cost more than they reveal.

## The scalability sweep

`d ∈ {2,…,8}` at `m ∈ {2,3}`, which takes the sweep well past d=6 — the ceiling of every public MORL
benchmark. MUSOLS's cost is roughly flat in d (it searches a 1–2 dimensional consensus space), while OLS grows
about 9× per added dimension.

**OLS is capped at 600 s per algorithm and will exhaust that from about d=6 onward.** That is the finding
rather than a failure, and `converged` records it per cell — but it means cost ratios from those cells are
**lower bounds**, since OLS never finished. Report them as censored, and read the converged fraction
alongside: it falls to zero exactly where the unrestricted search stops being viable.

## Reproducibility

Sharding selects *which* cells a process runs and never changes a cell's result — every random quantity is
derived from the cell's own coordinates, not from execution order. The union of shards is therefore exactly
what one serial run would have produced, and a failed shard can be resubmitted alone. Each record carries the
git commit, library versions and full hyperparameters, so a results file is self-describing.

# Running the MUSOLS study on SLURM

Full paper-scale sweep: **n=100 sampled stakeholder panels per configuration**, every environment, every
baseline. Roughly **3350 core-hours across 333 array tasks**. The deep-RL sweeps dominate that figure: at
500k-step training budgets, lunar-lander and highway alone account for about 2540 of those core-hours.

A separate job, `submit_demo.sbatch`, builds the showcase artifact for a talk or the paper's webpage. It is
*not* part of the evaluation — see "The showcase demo" below.

```bash
cd experiments/musols_benchmark/slurm
./submit_all.sh --dry-run      # print the plan and cost estimate, submit nothing
./submit_all.sh                # submit every sweep the paper reports
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
tasks share machines. `SDL_VIDEODRIVER=dummy` keeps highway-env's pygame dependency from trying to open a
display on a headless node.

**`MUJOCO_GL=disable`, not `osmesa`.** MuJoCo resolves its GL backend when the module is *imported*, not when
a frame is rendered, so `MUJOCO_GL=osmesa` makes `import mujoco` fail outright on a node without the OSMesa
shared library — PyOpenGL raises `'NoneType' object has no attribute 'glGetError'` — and it takes hopper and
**reacher** down with it even though neither renders anything during training. `disable` skips GL
initialization entirely, which is what a training-only job wants. Only a job that actually records video
needs a real backend: `submit_demo.sbatch` sets one only under `RECORD_VIDEO=1`.

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
| water-reservoir | 48600 | 75k measured at ~9700/cell, scaled to the 500k budget and the 21600 s cap |
| hopper (demo) | 300000+ | **measured**: 18780 s per 1M-step evaluation (job 27373892), x 16+ evaluations |

Most of these are linear extrapolations and their accuracy varies in both directions: minecart's real
per-evaluation cost came in at ~130–190 s against the 440 s its row predicted at 100k, while water-reservoir's
real per-cell cost came in at ~9700 s against a predicted 2384 s (4.1x *under*-estimated). Hopper's row was
revised after job 27373892 measured a full 1M-step evaluation on a Genoa core at **18780 s** against the 3658
s previously recorded here — 5.1x *under*-estimated, and the reason that job's 1800 s per-algorithm cap
allowed exactly one evaluation per algorithm. Treat every row in this table as a lower bound until a real run
on the target hardware has contradicted it.

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
- **hopper (demo only, 1M steps)**, `learning_starts=10_000`, `buffer_size=1_000_000`, `net_arch=[256,256]`,
  γ=0.99, autotune left **on**. The 3-link hopper terminates the instant the torso tilts too far, so early
  episodes are near-instant failures; 10k steps of uniform sampling gives the Q-function a spread of joint
  configurations before any gradient update, and the full 1M buffer keeps those early unstable steps
  comparable against steady-state strides acquired much later. Autotune is *kept* here, unlike highway:
  MOSAC starts alpha at exp(0)=1.0, which is the conservatively-high initial entropy this task wants so the
  policy does not commit early to a lopsided leg-extension pattern. Its reward scaler exists because
  `healthy_reward` (+1/step) is added to all three objectives, so an idle agent scores the *maximum* on the
  energy objective — standing still is a real local optimum, and the raw energy swing of 3/step is as large
  as forward velocity's.
- **water-reservoir (75k → 500k)**, plus `eval_episodes=50`. 75k was originally kept because policies
  visibly *differentiate* by weight vector there — but differentiation is far weaker than optimality, and
  optimality is what OLS and MUSOLS assume `solve(w)` returns. The study's own numbers show the cost of the
  gap: MUSOLS's Δ-EU is exactly 0 on every exact-solver environment and 1.45 here, and this is the one
  environment where the Random-Ω_W ablation matches it. The per-algorithm cap went 5400 → 21600 s with the
  budget: at ~450 s per evaluation at 75k, 500k puts an evaluation near 3000 s, and the old cap would have
  censored MUSOLS itself (it converged on 96.6% of cells at 75k, which is what makes its row readable). OLS
  remains censored at the cap, as it already was — 0/177 converged at 75k — and that is accepted.

Minecart is sparse-reward and even 500k may be low; if its policies still look untrained (a `|CCS|` of 1 for
every algorithm is the tell), that budget is the first thing to raise again.

### Grid differs by cost, on purpose

Cheap environments get the full `m ∈ {2,3,4} × κ ∈ {1,5,50}` sweep (900 cells) plus anytime trajectories. The
expensive deep-RL environments get `m ∈ {2,3}` at a single κ (200 cells) and no trajectories: n=100 panels per
configuration is what statistical power requires, and extra κ values there would cost more than they reveal.

## The showcase demo

`run_demo.py` answers a different question from `run_study.py`, for a different audience. The study asks
"does MUSOLS beat the baselines across many randomly sampled panels, with confidence intervals" — what a
reviewer needs. The demo asks what an audience needs: given **two named stakeholders with concrete stated
preferences**, what set of policies does each method actually hand them, and what does moving the consensus
between them do?

```bash
sbatch submit_demo.sbatch                      # hopper (default), ~50 h, one task
sbatch submit_demo.sbatch fruit-tree demo_ft   # any registered environment

# or locally, in seconds, against an exact solver -- useful for building the page:
python experiments/musols_benchmark/run_demo.py --env deep-sea-treasure --out demo_dst

# control environments can also render one rollout per coverage-set policy:
python experiments/musols_benchmark/run_demo.py --env hopper --record-video --out demo_hopper

# ...but after a cluster run, render from the checkpoints instead, on a machine with a GL backend:
python experiments/musols_benchmark/render_demo_videos.py demo/demo_hopper --deterministic
```

`render_demo_videos.py` exists because `--record-video` on a cluster node is a gamble that only pays out at
the very end of a multi-day run: MuJoCo resolves its GL backend at *import* time, so the training job sets
`MUJOCO_GL=disable` and a node without OSMesa cannot render at all. Rendering afterwards from the
checkpoints costs seconds, needs no rerun, and rewrites `video_files` in place so the directory is
indistinguishable from one produced inline. It reports each clip's discounted return next to the
`coverage_set` entry it should match, which is how you catch a checkpoint matched to the wrong policy.

Two environments are worth showing together. **deep-sea-treasure** is legible — a reader can see the whole
trade-off at once. **hopper** is the control example: d=3 (forward velocity, hop height, energy), continuous
actions, and with `--record-video` each coverage-set policy is rendered so an audience can watch the
logistics operator's gait next to the maintenance engineer's. Walker is *not* a substitute here: `mo-walker2d`
is only d=2 (velocity and control cost), so with m=2 stakeholders min(d,m)=2, Omega_W is the full simplex and
MUSOLS has nothing to restrict -- the demo would show no advantage by construction.

Three deliberate differences from the study: the panel is the environment's own hand-written
`stakeholder_weights` rather than a Dirichlet sample, so every number on screen belongs to a describable
scenario; policies are **checkpointed** so rollouts can be rendered per coverage-set member; and the
comparison is framed as *wasted work* — OLS searches the whole simplex, so some policies it trains are
optimal only for weights no consensus of these stakeholders can produce, and `run_demo.py` counts them by LP.

On deep-sea-treasure, for instance, MUSOLS returns 7 policies and OLS returns 9, of which **2 are optimal
only outside the consensus polytope** — MUSOLS's set is exactly OLS's useful subset.

### Noisy inner solvers break OLS's assumptions

OLS and MUSOLS are proved correct against an **exact** inner solver: `solve(w)` is assumed to return
`argmax_v w . v`. A deep-RL inner loop does not, and the error is one-sided in its consequences. A training
run that lands too *low* closes a corner weight that should have stayed open, and neither algorithm has any
mechanism that can reopen it — so one unlucky run silently truncates the search. Nothing is ever harmed by a
solve being too good, which is why the failure shows up as premature convergence rather than as noise.

MUSOLS is the more exposed of the two, and for exactly the reason it is worth having: making few solver calls
is the point, and few calls means no redundancy. Measured on hopper at a 40k-step budget, MUSOLS reported
`converged=True` after 2 evaluations with a single policy that was Pareto-dominated in all three objectives
by one OLS found, ~45% below it across the whole consensus segment. Replaying the same run with
`ExactEnumerationSolver` over OLS's payoffs recovers the intended result — MUSOLS returns exactly the 2
policies optimal in Omega_W, in 3 evaluations against OLS's 5 in 14 — so this is a property of the inner
solver, not a bug in the search.

Two solver knobs address two different noises, and they need opposite operators:

| knob | fixes | why |
|---|---|---|
| `num_seeds` | training sometimes lands badly | best of k independent runs, selected by **max** of `w . v` — the inner problem is a maximization, so a mean estimates the wrong quantity |
| `eval_episodes` | finite-episode Monte Carlo return estimates | those estimates *are* the geometry corner weights are computed from; rollouts are a rounding error against training |

`solve_best_of_seeds` in `solvers.py` deliberately separates selection from reporting: taking the max over
noisy estimates would trade a pessimistic bias for an optimistic one, picking whichever run got lucky in
*evaluation*. That is the worse failure, because an over-estimated vector enters the coverage set as a value
no policy can achieve and can dominate and permanently delete genuine members. So the winner is chosen on one
evaluation and then re-evaluated on fresh episodes for the value actually returned.

Both default to the study's original behaviour (`num_seeds=1`, `eval_episodes=5`) so existing results stay
comparable. `submit_demo.sbatch` raises them, since a demo is a handful of solver calls whose failure is
unrecoverable rather than 100 panels whose noise averages out.

**Robust variants.** `run_experiment.ROBUST_VARIANTS` holds three relaxations of the exact-solver
assumption, each run and recorded as its own algorithm (`--robust-variants ...` in `run_study.py`) so plain
MUSOLS's numbers are never overwritten and the classical rule can be reported as the ablation:

| variant | rule |
|---|---|
| `musols_mono` | discard a solve that is beaten at its own weight by a vector already found — under an exact solver it cannot be an argmax, so this cannot occur |
| `musols_reopen` | reopen a weight once the run has *certified* its own solve suboptimal, i.e. some known vector beats what the solver achieved there |
| `musols_opt` | keep a solved weight eligible on *suspicion*: `max_value_lp` stops capping the optimistic bound at that weight's own, possibly underestimated, recorded value |

Measured on a simulation over the 177 real water-reservoir panels, with the inner solver replaced by one that
returns a worse candidate with probability `p` (mean EU shortfall against the best achievable set):

| variant | p=0 evals | p=0 shortfall | p=0.5 evals | p=0.5 shortfall |
|---|---|---|---|---|
| MUSOLS (classical) | 3.78 | 0 | 3.38 | 1.358 |
| `musols_reopen` | 3.78 | 0 | 3.68 | 1.348 |
| `musols_opt`, `max_resolves=1` | 7.42 | 0 | 7.38 | 0.247 |
| `musols_opt`, `max_resolves=2` | 11.07 | 0 | 11.23 | 0.118 |
| `musols_opt`, `max_resolves=3` | 14.71 | 0 | 15.09 | <0.001 |

Three things to read off it. **Only the optimistic variant works**: the certificate is sound but nearly inert,
because it can fire only once a better policy has turned up at some *other* weight — circular exactly when the
search is short enough that one bad solve truncates it. **Nothing is lost when the solver is exact** (shortfall
stays 0), so the variants are safe, but they are not free: `musols_opt` costs 2–4x the evaluations regardless
of whether the noise it guards against is present. **`max_resolves` is the real knob**, not the size of the
slack — any positive slack makes a weight eligible and the cap decides the spend.

That cost lands directly on MUSOLS's headline claim. OLS averages 10.9 evaluations on water-reservoir, so
`max_resolves=2` draws level with it and `max_resolves=3` is dearer; only `max_resolves=1` (2x the
evaluations for a 5.5x reduction in shortfall) keeps a meaningful efficiency advantage, which is why it is
the registered default. Caveat on the simulation: re-solves draw from a fixed pool in which the good policy
is always findable, so it flatters the variants relative to real RL, where a re-solve may simply fail again.

**Consequences for what the paper can claim.** With an approximate inner solver, OLS's guarantees do not
hold — there is theory for approximate variants giving a bounded-quality result under a uniformly
epsilon-optimal subroutine, but SAC at a fixed step budget satisfies no such bound. The algorithmic claim
therefore rests on the exact-solver sweeps (deep-sea-treasure, fruit-tree, synthetic); the deep-RL
environments demonstrate applicability, not optimality, and should be reported that way. Note also that
MUSOLS and OLS make different numbers of solver calls and so have different exposure to this noise, which
confounds a head-to-head on any deep-RL environment: a MUSOLS loss there may be solver variance rather than
search quality. Holding the *total training budget* fixed rather than the number of distinct weights is the
cleaner comparison, and it lets MUSOLS spend its smaller weight count on more seeds per weight.

### The artifact

`demo.json` is self-contained and carries `W` plus both coverage sets, which is everything needed to drive an
interactive consensus slider **client-side**: for a consensus `alpha` in the simplex, the selected policy is
`argmax_v (W @ alpha) . v` over that algorithm's coverage set. No server, no model inference. For m=2 a
precomputed `consensus_sweep` is included as well, so a page can bind a slider directly to it.

| key | what it holds |
|---|---|
| `W`, `objective_names`, `stakeholder_names` | the scenario, ready to label a UI |
| `algorithms.{musols,ols}.coverage_set` | the payoff vectors each method returned |
| `algorithms.*.useful_for_consensus` | per policy: is it optimal anywhere in Omega_W? |
| `algorithms.*.policy_files` | checkpoint per coverage-set member, for rendering rollouts |
| `consensus_sweep` | alpha -> winning policy, precomputed (m=2 only) |
| `comparison` | wasted-policy count, evaluation ratio, wall-clock speedup |

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

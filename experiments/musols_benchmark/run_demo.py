"""Builds the showcase demo: one realistic stakeholder panel, MUSOLS vs OLS, with policies kept.

This is not the evaluation. `run_study.py` answers "does MUSOLS beat the baselines across many randomly
sampled panels, with confidence intervals" -- the question a reviewer asks. This script answers the question
an *audience* asks, in a talk or on the paper's webpage: given two named stakeholders with concrete, stated
preferences, what set of policies does each method actually hand them, how long did each take, and what does
moving the consensus between the two stakeholders actually do?

So it differs from the study in three deliberate ways:

  * **One fixed, realistic panel, not a sampled one.** W comes from the environment's own hand-written
    `stakeholder_weights` (e.g. hopper's "logistics operator" vs "maintenance engineer"), so every number on
    screen belongs to a scenario that can be described in a sentence. No Dirichlet sampling, no seed sweep.
  * **Policies are kept.** Each coverage-set member is checkpointed, so rollouts can be rendered per policy
    and a viewer can watch what "the maintenance engineer's preferred gait" actually looks like.
  * **The comparison is framed as wasted work.** OLS searches the whole simplex, so some of the policies it
    trains are optimal only for weight vectors no consensus of *these* stakeholders can produce. Those are
    counted explicitly: they are the concrete cost of not knowing the preferences up front.

Output is a single self-describing `demo.json` plus policy checkpoints. The JSON carries W and the coverage
sets, which is everything needed to resolve the interactive question client-side: for a consensus
`alpha` in the simplex, the chosen policy is `argmax_v (W @ alpha) . v` over that algorithm's coverage set --
no server and no model inference required to drive a slider.

Examples:
    # Cheap, exact, runs in seconds -- good for building the page against real output.
    python experiments/musols_benchmark/run_demo.py --env deep-sea-treasure --out demo_dst

    # The real thing (expensive; use the SLURM wrapper).
    python experiments/musols_benchmark/run_demo.py --env hopper --total-timesteps 1000000 --out demo_hopper
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch as th
from environments import ENVIRONMENTS
from outer_loop import run_outer_loop
from run_experiment import _make_seeded_env
from scipy.optimize import linprog
from solvers import make_solver

from morl_baselines.common.evaluation import seed_everything
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.multi_policy.linear_support.musols import MUSOLS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, default="hopper", help="Registered environment key.")
    parser.add_argument("--out", type=str, required=True, help="Directory for demo.json and policy checkpoints.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override the RL training budget.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-algorithm wall-clock cap (s).")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=("exact", "tabular_q", "sac", "discrete_sac"),
        help=(
            "Override the environment's inner-loop solver. The point of a demo is usually to show *learned* "
            "policies, so an environment the study solves by exact enumeration (deep-sea-treasure, fruit-tree) "
            "has no policy to roll out or render -- pass tabular_q there to train real ones instead."
        ),
    )
    parser.add_argument(
        "--solver-kwargs",
        type=str,
        default=None,
        help='JSON dict merged into the solver hyperparameters, e.g. \'{"final_epsilon":0.02}\'.',
    )
    parser.add_argument(
        "--alpha-samples",
        type=int,
        default=201,
        help="Resolution of the precomputed consensus sweep written into demo.json.",
    )
    parser.add_argument("--skip-policies", action="store_true", help="Do not checkpoint policies (JSON only).")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Render one rollout per coverage-set policy to an mp4. Needs a renderable environment.",
    )
    parser.add_argument("--video-steps", type=int, default=500, help="Max steps per recorded rollout.")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help=(
            "Frame rate for recorded rollouts. 30 suits a MuJoCo episode of hundreds of steps; a gridworld "
            "whose whole episode is a dozen steps needs ~3, or the clip is over before it can be watched."
        ),
    )
    return parser.parse_args()


def record_rollout(config, agent, path, max_steps, fps=30.0):
    """Renders one greedy rollout of `agent` to an mp4, for showing what a coverage-set policy actually does.

    Returns the written path, or None if the environment cannot render (exact-solver environments have no
    policy to roll out, and some environments have no rgb_array mode). Rendering is strictly optional: a
    failure here must never lose the demo's numeric artifact, so every error is swallowed.
    """
    import imageio
    import mo_gymnasium as mo_gym

    try:
        env = mo_gym.make(config.env_id, render_mode="rgb_array", **config.env_kwargs)
        if config.observation_wrapper is not None:
            env = config.observation_wrapper(env)
        frames = []
        obs, _ = env.reset(seed=0)
        for _ in range(max_steps):
            action = agent.eval(np.asarray(obs, dtype=np.float32), w=agent.weights)
            obs, _, term, trunc, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if term or trunc:
                break
        env.close()
        if not frames:
            return None
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception as exc:  # rendering is a nice-to-have; never fail the run over it
        print(f"    (video skipped: {type(exc).__name__}: {exc})")
        return None


def is_optimal_somewhere(payoff, coverage_set, weight_polytope_columns=None):
    """True if `payoff` is the strict argmax over `coverage_set` for at least one admissible weight.

    With `weight_polytope_columns = W`, admissible weights are restricted to the consensus polytope
    Omega_W = {W alpha : alpha in simplex}; the LP is then solved over alpha rather than over w. This is what
    makes "policies OLS trained that no consensus of these stakeholders can ever prefer" a computable number
    rather than an intuition.
    """
    others = [np.asarray(v, float) for v in coverage_set if not np.allclose(v, payoff)]
    if not others:
        return True
    payoff = np.asarray(payoff, float)
    diffs = np.array([payoff - o for o in others])          # rows: (v - v_j), want all . w >= eps
    if weight_polytope_columns is None:
        n_var = len(payoff)
        A_ub, A_eq = -diffs, np.ones((1, n_var))
    else:
        W = np.asarray(weight_polytope_columns, float)      # (d, m); w = W @ alpha
        n_var = W.shape[1]
        A_ub, A_eq = -(diffs @ W), np.ones((1, n_var))
    res = linprog(
        c=np.zeros(n_var),
        A_ub=A_ub,
        b_ub=-1e-9 * np.ones(len(others)),
        A_eq=A_eq,
        b_eq=[1.0],
        bounds=[(0, 1)] * n_var,
        method="highs",
    )
    return bool(res.success)


def consensus_sweep(coverage_set, W, num_samples):
    """Which coverage-set member wins as the consensus slides between stakeholders.

    Only emitted for m = 2, where the consensus is a single number (alpha in [0, 1]) and therefore drivable by
    one slider. For m > 2 the page can still compute the argmax itself from W and the coverage set.
    """
    if W.shape[1] != 2 or not coverage_set:
        return None
    cs = np.array([np.asarray(v, float) for v in coverage_set])
    out = []
    for a in np.linspace(0.0, 1.0, num_samples):
        w = W @ np.array([1.0 - a, a])
        scores = cs @ w
        idx = int(np.argmax(scores))
        out.append({"alpha": round(float(a), 6), "policy_index": idx, "utility": float(scores[idx])})
    return out


def run_algorithm(name, config, args, W, out_dir):
    """Runs one algorithm on the fixed panel and checkpoints the policy behind each coverage-set member."""
    solver_kwargs = dict(config.solver_kwargs)
    if args.solver is not None and args.solver != config.solver:
        # Switching solver invalidates the registry's hyperparameters, which were tuned for the original one.
        solver_kwargs = {}
    if args.total_timesteps is not None:
        solver_kwargs["total_timesteps"] = args.total_timesteps
    if args.solver_kwargs:
        solver_kwargs.update(json.loads(args.solver_kwargs))
    epsilon = config.epsilon if args.epsilon is None else args.epsilon
    timeout = config.timeout_seconds if args.timeout is None else args.timeout

    seed_everything(args.seed)
    env = _make_seeded_env(config, args.seed)
    solver = make_solver(config, env, args.seed, solver_kwargs)

    # Checkpoint whatever policy the solver just trained, keyed by the payoff it achieved, so coverage-set
    # members can be matched back to a policy after the outer loop has finished pruning dominated ones.
    trained = {}
    inner_solve = solver.solve

    def recording_solve(w):
        value = inner_solve(w)
        agent = getattr(solver, "last_agent", None)
        if agent is not None and not args.skip_policies:
            trained[tuple(np.round(np.asarray(value, float), 6))] = agent
        return value

    solver.solve = recording_solve

    algo = (
        MUSOLS(user_weights=W, epsilon=epsilon, verbose=False)
        if name == "musols"
        else LinearSupport(num_objectives=config.num_objectives, epsilon=epsilon, verbose=False)
    )
    result = run_outer_loop(algo, solver, max_seconds=timeout)
    env.close()

    ccs = [np.asarray(v, float) for v in result.ccs]
    record = {
        "coverage_set": [v.tolist() for v in ccs],
        "num_evaluated": result.num_evaluated,
        "elapsed_seconds": result.elapsed_seconds,
        "converged": bool(result.converged),
        "weight_support": [np.asarray(w, float).tolist() for w in result.weight_support],
        "policy_files": [],
    }
    if name == "musols":
        record["consensus_weight_support"] = [a.tolist() for a in algo.get_consensus_weight_support()]

    # Every MUSOLS policy is by construction optimal somewhere in Omega_W. For OLS this is the headline demo
    # number: how much of its work these particular stakeholders can never use.
    record["useful_for_consensus"] = [bool(is_optimal_somewhere(v, ccs, W)) for v in ccs]

    record["video_files"] = []
    if not args.skip_policies:
        policy_dir = out_dir / "policies" / name
        policy_dir.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(ccs):
            agent = trained.get(tuple(np.round(v, 6)))
            if agent is None or not hasattr(agent, "get_save_dict"):
                record["policy_files"].append(None)
                record["video_files"].append(None)
                continue
            path = policy_dir / f"policy_{i:02d}.pt"
            th.save(agent.get_save_dict(save_replay_buffer=False), path)
            record["policy_files"].append(str(path.relative_to(out_dir)))

            video = None
            if args.record_video:
                video_dir = out_dir / "videos" / name
                video_dir.mkdir(parents=True, exist_ok=True)
                written = record_rollout(
                    config, agent, video_dir / f"policy_{i:02d}.mp4", args.video_steps, args.video_fps
                )
                video = str(Path(written).relative_to(out_dir)) if written else None
            record["video_files"].append(video)
    return record


def main():
    args = parse_args()
    assert args.env in ENVIRONMENTS, f"Unknown env {args.env!r}. Known: {sorted(ENVIRONMENTS)}"
    config = ENVIRONMENTS[args.env]
    if args.solver is not None and args.solver != config.solver:
        config = replace(config, solver=args.solver)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The realistic panel: the environment's own named stakeholders, not a sampled one.
    W = np.asarray(config.stakeholder_weights, dtype=np.float32)

    print(f"Demo: {config.key}  ({config.num_objectives} objectives, {W.shape[1]} stakeholders)")
    for i, who in enumerate(config.stakeholder_names):
        prefs = ", ".join(f"{n}={W[j, i]:.2f}" for j, n in enumerate(config.objective_names))
        print(f"  {who}: {prefs}")

    results = {}
    for name in ("musols", "ols"):
        print(f"\nrunning {name} ...", flush=True)
        results[name] = run_algorithm(name, config, args, W, out_dir)
        r = results[name]
        print(
            f"  |CCS|={len(r['coverage_set'])}  evals={r['num_evaluated']}  "
            f"{r['elapsed_seconds']:.1f}s  converged={r['converged']}"
        )

    ols_useful = sum(results["ols"]["useful_for_consensus"])
    ols_total = len(results["ols"]["coverage_set"])
    musols_t = max(results["musols"]["elapsed_seconds"], 1e-9)

    demo = {
        "env": config.key,
        "description": config.description,
        "objective_names": config.objective_names,
        "stakeholder_names": config.stakeholder_names,
        # Column i is stakeholder i's weight vector; a consensus weight is W @ alpha for alpha in the simplex.
        "W": W.tolist(),
        "gamma": config.gamma,
        "solver": config.solver,
        "solver_is_learned": config.solver != "exact",
        "algorithms": results,
        "comparison": {
            "ols_policies_total": ols_total,
            "ols_policies_useful_for_consensus": ols_useful,
            "ols_policies_wasted": ols_total - ols_useful,
            "musols_policies": len(results["musols"]["coverage_set"]),
            "evaluation_ratio": results["ols"]["num_evaluated"] / max(results["musols"]["num_evaluated"], 1),
            "speedup": results["ols"]["elapsed_seconds"] / musols_t,
        },
        # Precomputed so a page can drive a slider with no maths of its own; only meaningful for m = 2.
        "consensus_sweep": {
            name: consensus_sweep(r["coverage_set"], W, args.alpha_samples) for name, r in results.items()
        },
        "provenance": {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": th.__version__,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps or config.solver_kwargs.get("total_timesteps"),
        },
    }
    (out_dir / "demo.json").write_text(json.dumps(demo, indent=2))

    c = demo["comparison"]
    print(f"\n{'':-<70}")
    print(f"MUSOLS returned {c['musols_policies']} polic(ies) in {results['musols']['elapsed_seconds']:.1f}s "
          f"({results['musols']['num_evaluated']} evaluations)")
    print(f"OLS    returned {c['ols_policies_total']} polic(ies) in {results['ols']['elapsed_seconds']:.1f}s "
          f"({results['ols']['num_evaluated']} evaluations)")
    print(f"  of which {c['ols_policies_wasted']} are optimal ONLY outside this panel's consensus polytope")
    print(f"  -> {c['speedup']:.1f}x wall-clock, {c['evaluation_ratio']:.1f}x the policy trainings")
    print(f"\nWrote {out_dir / 'demo.json'}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())

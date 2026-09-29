"""Runs a single MUSOLS-vs-OLS benchmark experiment on one registered environment.

Parameterizes the environment, seed, inner-loop solver hyperparameters, and per-algorithm wall-clock timeout,
so a single experiment can be reproduced or re-tuned entirely from the command line.

Examples:
    python experiments/musols_benchmark/run_experiment.py --env fruit-tree --seed 0
    python experiments/musols_benchmark/run_experiment.py --env lunar-lander --seed 1 --total-timesteps 20000 --timeout 1200
    python experiments/musols_benchmark/run_experiment.py --env resource-gathering --skip-ols
"""

import argparse

import mo_gymnasium as mo_gym
from baselines import RandomOmegaWSearch, VertexOnlySearch
from environments import ENVIRONMENTS
from outer_loop import OuterLoopResult, run_outer_loop
from solvers import make_solver

from morl_baselines.common.evaluation import seed_everything
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.multi_policy.linear_support.musols import MUSOLS


def _make_seeded_env(config, seed: int):
    """Creates a fresh environment instance, applies its reward normalization (if any), and fully seeds it
    (state, action space, observation space).

    `env.reset(seed=seed)` alone does not seed `env.action_space`: it is a separate RNG (lazily created with
    OS entropy if never explicitly seeded) that RL solvers use for exploratory `action_space.sample()` calls.
    Leaving it unseeded silently breaks reproducibility for the tabular_q/sac solvers, even with a fixed seed
    everywhere else.
    """
    env = mo_gym.make(config.env_id, **config.env_kwargs)
    if config.observation_wrapper is not None:
        env = config.observation_wrapper(env)
    if config.reward_wrapper is not None:
        env = config.reward_wrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, choices=sorted(ENVIRONMENTS), required=True, help="Registered environment key.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, for full reproducibility.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override the environment's default OLS/MUSOLS epsilon.")
    parser.add_argument(
        "--timeout", type=float, default=None, help="Override the per-algorithm wall-clock budget, in seconds."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the RL solver's per-policy training budget (ignored for exact solvers).",
    )
    parser.add_argument(
        "--skip-ols",
        action="store_true",
        help="Skip the full-simplex OLS baseline and only run MUSOLS (useful for slow/expensive environments).",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip the Random-Omega_W baseline (uniform weight sampling restricted to the same polytope MUSOLS "
        "searches, given the same wall-clock budget MUSOLS used -- isolates the value of MUSOLS's search "
        "strategy from the value of restricting to Omega_W at all; see baselines.py).",
    )
    parser.add_argument(
        "--skip-vertex",
        action="store_true",
        help="Skip the vertex-only baseline (solve each stakeholder's own preference and stop; isolates "
        "the value of covering the consensus interior rather than only the individual optima).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress/result printing.")
    return parser.parse_args()


def run_experiment(
    env_key: str,
    seed: int = 0,
    epsilon: float = None,
    timeout: float = None,
    total_timesteps: int = None,
    skip_ols: bool = False,
    skip_random: bool = False,
    skip_vertex: bool = False,
    record_trajectory: bool = False,
    user_weights=None,
    verbose: bool = True,
) -> dict:
    """Runs the MUSOLS-vs-baselines benchmark for a single registered environment.

    Args:
        env_key: Key into `environments.ENVIRONMENTS`.
        seed: Random seed. The same seed reproduces the same environment dynamics, solver training and
            outer-loop trajectory for MUSOLS and every baseline.
        epsilon: Overrides the environment's default OLS/MUSOLS epsilon, if given.
        timeout: Overrides the environment's default per-algorithm wall-clock budget (seconds), if given. Only
            applies to OLS: the Random-Omega_W baseline instead gets exactly the wall-clock time MUSOLS used
            (see below).
        total_timesteps: Overrides the RL solver's per-policy training budget, if given (exact solvers ignore
            this).
        skip_ols: If True, skip the full-simplex OLS baseline.
        skip_random: If True, skip the Random-Omega_W baseline.
        skip_vertex: If True, skip the vertex-only (per-stakeholder optimum) baseline.
        record_trajectory: If True, have every algorithm snapshot its coverage set after each iteration,
            so quality can be plotted against evaluations spent (anytime curves).
        user_weights: Overrides the environment's hand-picked stakeholder preference matrix W, if given (shape
            (num_objectives, m)). Used by `run_study.py` to sweep randomly sampled panels of stakeholders.
        verbose: If True, prints progress and a result summary.

    Returns:
        A dict with key "musols" and (unless skipped) "random", "vertex" and/or "ols", mapping to
        `OuterLoopResult`s.
    """
    config = ENVIRONMENTS[env_key]
    epsilon = config.epsilon if epsilon is None else epsilon
    timeout = config.timeout_seconds if timeout is None else timeout
    weights = config.stakeholder_weights if user_weights is None else user_weights
    solver_kwargs = dict(config.solver_kwargs)
    if total_timesteps is not None:
        solver_kwargs["total_timesteps"] = total_timesteps

    if verbose:
        print(f"=== {config.key}: {config.description}")
        print(
            f"    num_objectives={config.num_objectives} ({', '.join(config.objective_names)}), "
            f"num_users={weights.shape[1]}, solver={config.solver}, seed={seed}"
        )
        if config.reward_wrapper is not None:
            print(f"    NOTE: rewards are normalized via {config.reward_wrapper.__name__} (see reward_normalization.py)")

    results = {}

    seed_everything(seed)
    musols_env = _make_seeded_env(config, seed)
    musols_solver = make_solver(config, musols_env, seed, solver_kwargs)
    musols = MUSOLS(user_weights=weights, epsilon=epsilon, verbose=False)
    results["musols"] = run_outer_loop(musols, musols_solver, max_seconds=timeout, record_trajectory=record_trajectory)
    musols_env.close()

    if not skip_random:
        # Re-seed everything identically so the baseline sees the same environment dynamics and solver
        # stochasticity that MUSOLS did, then give it exactly as many inner-loop evaluations as MUSOLS used.
        # Evaluation-count parity (rather than wall-clock parity) equalizes the genuinely expensive resource,
        # and keeps the comparison exactly reproducible: a wall-clock budget would let machine load change how
        # many samples the baseline fits in. This isolates the value of MUSOLS's search strategy, at equal
        # budget, from the value of merely restricting the search to Omega_W (see baselines.py).
        seed_everything(seed)
        random_env = _make_seeded_env(config, seed)
        random_solver = make_solver(config, random_env, seed, solver_kwargs)
        random_algo = RandomOmegaWSearch(user_weights=weights, seed=seed, verbose=False)
        results["random"] = run_outer_loop(
            random_algo,
            random_solver,
            max_seconds=timeout,
            max_evaluations=results["musols"].num_evaluated,
            record_trajectory=record_trajectory,
        )
        random_env.close()

    if not skip_vertex:
        # The naive baseline: solve each stakeholder's own preference and stop. It self-terminates after m
        # evaluations, so it needs no budget beyond the environment's safety timeout.
        seed_everything(seed)
        vertex_env = _make_seeded_env(config, seed)
        vertex_solver = make_solver(config, vertex_env, seed, solver_kwargs)
        vertex_algo = VertexOnlySearch(user_weights=weights, verbose=False)
        results["vertex"] = run_outer_loop(
            vertex_algo, vertex_solver, max_seconds=timeout, record_trajectory=record_trajectory
        )
        vertex_env.close()

    if not skip_ols:
        # Re-seed everything identically so OLS sees the same environment dynamics and solver stochasticity
        # that MUSOLS did, making the comparison fair and the whole experiment reproducible end to end.
        seed_everything(seed)
        ols_env = _make_seeded_env(config, seed)
        ols_solver = make_solver(config, ols_env, seed, solver_kwargs)
        ols = LinearSupport(num_objectives=config.num_objectives, epsilon=epsilon, verbose=False)
        results["ols"] = run_outer_loop(ols, ols_solver, max_seconds=timeout, record_trajectory=record_trajectory)
        ols_env.close()

    if verbose:
        _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    def line(name: str, result: OuterLoopResult, anytime: bool = False) -> str:
        if anytime:
            status = " (anytime baseline, ran for its full budget)"
        else:
            status = "" if result.converged else " (did NOT converge within the time budget)"
        return (
            f"    {name:<7}: {len(result.ccs)} in CCS, {result.num_evaluated} evaluated, {result.elapsed_seconds:.2f}s{status}"
        )

    print(line("MUSOLS", results["musols"]))
    if "random" in results:
        print(line("Random", results["random"], anytime=True))
    if "vertex" in results:
        print(line("Vertex", results["vertex"]))
    if "ols" in results:
        print(line("OLS", results["ols"]))
    print()


def main():
    args = parse_args()
    run_experiment(
        env_key=args.env,
        seed=args.seed,
        epsilon=args.epsilon,
        timeout=args.timeout,
        total_timesteps=args.total_timesteps,
        skip_ols=args.skip_ols,
        skip_random=args.skip_random,
        skip_vertex=args.skip_vertex,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

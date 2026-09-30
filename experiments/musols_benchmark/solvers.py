"""Pluggable inner-loop solvers for the OLS/MUSOLS outer loop.

Each solver implements `solve(w) -> np.ndarray`: given an objective weight vector w, it returns the (exact or
learned) discounted vector return of the best policy found for that weight. Which solver to use for a given
environment is declared in `environments.py`.
"""

import numpy as np

from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning
from morl_baselines.single_policy.ser.mosac_continuous_action import MOSAC
from morl_baselines.single_policy.ser.mosac_discrete_action import MOSACDiscrete


def solve_best_of_seeds(w, num_seeds, eval_episodes, eval_env, train_one):
    """Trains `num_seeds` independent policies for weight `w` and returns the best one, with a clean value.

    OLS and MUSOLS assume an *exact* inner solver: `solve(w)` is supposed to return `argmax_v w . v`. A
    noisy RL solver breaks that assumption one-sidedly. A solve that comes back too *low* closes a corner
    weight that should have stayed open, and OLS has no mechanism that can ever reopen it -- so a single
    unlucky training run silently truncates the search. (Observed on hopper: MUSOLS reported converged with
    one policy that was Pareto-dominated by one OLS found.) MUSOLS is the more exposed of the two precisely
    because its whole point is to make few solver calls, so it has no redundancy to absorb a bad one.

    Two distinct noises need opposite treatment, which is why this takes both parameters:

      * **Policy quality** -- training sometimes lands badly. The remedy is `num_seeds` retries combined with
        **max**, not mean: the inner problem is a maximization, and the mean estimates "what a typical run
        achieves at w", which is not the quantity the outer loop needs.
      * **Value estimation** -- a finite-episode Monte Carlo estimate of a stochastic policy's return. The
        remedy is `eval_episodes`, and it is cheap: rollouts cost a rounding error against training.

    Selecting the max over noisy estimates would trade the pessimistic bias for an optimistic one -- picking
    whichever run got *lucky in evaluation* rather than whichever policy is best. That is the worse failure,
    because an over-estimated vector enters the coverage set as a value no policy can actually achieve, and
    can dominate and permanently delete genuine members. So selection and reporting are separated: the
    winner is chosen on one evaluation and then **re-evaluated on fresh episodes** for the value returned.

    Args:
        w: The objective weight vector being solved for.
        num_seeds: Independent training runs to take the best of. 1 reproduces the single-shot behaviour.
        eval_episodes: Episodes per evaluation, for both the selection and the final re-evaluation.
        eval_env: Environment to evaluate on.
        train_one: Callable `seed_offset -> trained agent`.

    Returns:
        Tuple of (best agent, its discounted vector return under a fresh evaluation).
    """
    best_agent, best_score = None, -np.inf
    for k in range(num_seeds):
        agent = train_one(k)
        _, _, _, value = agent.policy_eval(eval_env=eval_env, weights=w, num_episodes=eval_episodes)
        score = float(np.dot(w, value))
        if score > best_score:
            best_agent, best_score = agent, score

    # Fresh episodes, so the reported vector is not the statistic the winner was selected on.
    _, _, _, value = best_agent.policy_eval(eval_env=eval_env, weights=w, num_episodes=eval_episodes)
    return best_agent, value


class ExactEnumerationSolver:
    """Exact inner-loop solver for small, deterministic, fully enumerable environments.

    Solves by argmax over a fixed, precomputed set of achievable payoff vectors (e.g. an environment's known
    analytical Pareto front). Deterministic: the same weight always yields the same value.
    """

    def __init__(self, candidates: np.ndarray):
        self.candidates = np.asarray(candidates, dtype=np.float32)

    def solve(self, w: np.ndarray) -> np.ndarray:
        return self.candidates[np.argmax(self.candidates @ w)]


class TabularQLearningSolver:
    """Trains a real tabular Q-learning policy (MOQLearning) per candidate weight.

    Suited to small, discrete-observation environments. Each call to `solve` trains and evaluates a fresh
    policy; successive calls use a deterministically incrementing seed, derived from `seed`, for reproducibility.
    """

    def __init__(
        self,
        env,
        gamma: float,
        seed: int,
        total_timesteps: int = 30_000,
        learning_rate: float = 0.3,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        epsilon_decay_frac: float = 0.5,
        num_seeds: int = 1,
        eval_episodes: int = 5,
    ):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_frac = epsilon_decay_frac
        # Both default to the single-shot behaviour the study was run with, so existing results stay
        # comparable; raise them per run via solver_kwargs. See solve_best_of_seeds for what each one fixes.
        self.num_seeds = int(num_seeds)
        self.eval_episodes = int(eval_episodes)
        self._num_solved = 0

    def solve(self, w: np.ndarray) -> np.ndarray:
        def train_one(seed_offset):
            agent = MOQLearning(
                self.env,
                weights=w,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                initial_epsilon=self.initial_epsilon,
                final_epsilon=self.final_epsilon,
                epsilon_decay_steps=int(self.total_timesteps * self.epsilon_decay_frac),
                seed=self.seed + self._num_solved + seed_offset,
                log=False,
            )
            agent.train(0, total_timesteps=self.total_timesteps)
            return agent

        agent, discounted_return = solve_best_of_seeds(
            w, self.num_seeds, self.eval_episodes, self.env, train_one
        )
        self._num_solved += self.num_seeds
        self.last_agent = agent  # retained so demos can checkpoint the policy; unused by the study path
        return discounted_return


class ContinuousSACSolver:
    """Trains a real continuous-control SAC policy (MOSAC) per candidate weight.

    Suited to continuous-observation, continuous-action environments. Each call to `solve` trains and
    evaluates a fresh policy; successive calls use a deterministically incrementing seed, derived from `seed`,
    for reproducibility.
    """

    def __init__(
        self,
        env,
        gamma: float,
        seed: int,
        total_timesteps: int = 8_000,
        net_arch=(64, 64),
        learning_starts: int = 256,
        batch_size: int = 128,
        buffer_size: int = None,
        alpha: float = 0.2,
        autotune: bool = True,
        num_seeds: int = 1,
        eval_episodes: int = 5,
    ):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.net_arch = list(net_arch)
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        # MOSAC preallocates the whole buffer eagerly, so sizing it to the run rather than leaving it at the
        # 1e6 default avoids reserving memory for transitions this run can never collect.
        self.buffer_size = int(buffer_size) if buffer_size is not None else min(total_timesteps, 1_000_000)
        self.alpha = alpha
        self.autotune = autotune
        # Both default to the single-shot behaviour the study was run with, so existing results stay
        # comparable; raise them per run via solver_kwargs. See solve_best_of_seeds for what each one fixes.
        self.num_seeds = int(num_seeds)
        self.eval_episodes = int(eval_episodes)
        self._num_solved = 0

    def solve(self, w: np.ndarray) -> np.ndarray:
        def train_one(seed_offset):
            agent = MOSAC(
                self.env,
                weights=w,
                gamma=self.gamma,
                net_arch=self.net_arch,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                alpha=self.alpha,
                autotune=self.autotune,
                seed=self.seed + self._num_solved + seed_offset,
                log=False,
            )
            agent.train(total_timesteps=self.total_timesteps, eval_env=self.env)
            return agent

        agent, discounted_return = solve_best_of_seeds(
            w, self.num_seeds, self.eval_episodes, self.env, train_one
        )
        self._num_solved += self.num_seeds
        self.last_agent = agent  # retained so demos can checkpoint the policy; unused by the study path
        return discounted_return


class DiscreteSACSolver:
    """Trains a real discrete-action SAC policy (MOSACDiscrete) per candidate weight.

    Suited to environments with a continuous observation space but a discrete action space, where neither the
    tabular solver (no table over continuous observations) nor the continuous-action solver (which requires a
    Box action space) applies. Each call to `solve` trains and evaluates a fresh policy; successive calls use a
    deterministically incrementing seed, derived from `seed`, for reproducibility.
    """

    def __init__(
        self,
        env,
        gamma: float,
        seed: int,
        total_timesteps: int = 50_000,
        net_arch=(64, 64),
        learning_starts: int = 1_000,
        batch_size: int = 128,
        update_frequency: int = 4,
        target_net_freq: int = 200,
        buffer_size: int = None,
        alpha: float = 0.2,
        autotune: bool = True,
        num_seeds: int = 1,
        eval_episodes: int = 5,
    ):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.net_arch = list(net_arch)
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        # MOSACDiscrete preallocates the whole buffer eagerly; size it to the run unless told otherwise.
        self.buffer_size = int(buffer_size) if buffer_size is not None else min(total_timesteps, 1_000_000)
        # NOTE: MOSACDiscrete ignores `alpha` whenever `autotune` is True -- it initializes log_alpha to zeros,
        # so the entropy coefficient *starts at exp(0) = 1.0* regardless of what is passed here. Setting a
        # deliberately conservative initial alpha therefore requires autotune=False (see environments.py's
        # highway config, where an alpha of 1.0 means near-uniform random lane changes early in training).
        self.alpha = alpha
        self.autotune = autotune
        self.update_frequency = update_frequency
        # MOSACDiscrete asserts this divisibility; check here so a bad config fails at construction with a
        # clear message rather than deep inside the first training run.
        assert target_net_freq % update_frequency == 0, "target_net_freq must be divisible by update_frequency."
        self.target_net_freq = target_net_freq
        # Both default to the single-shot behaviour the study was run with, so existing results stay
        # comparable; raise them per run via solver_kwargs. See solve_best_of_seeds for what each one fixes.
        self.num_seeds = int(num_seeds)
        self.eval_episodes = int(eval_episodes)
        self._num_solved = 0

    def solve(self, w: np.ndarray) -> np.ndarray:
        def train_one(seed_offset):
            agent = MOSACDiscrete(
                self.env,
                weights=w,
                gamma=self.gamma,
                net_arch=self.net_arch,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                update_frequency=self.update_frequency,
                target_net_freq=self.target_net_freq,
                buffer_size=self.buffer_size,
                alpha=self.alpha,
                autotune=self.autotune,
                seed=self.seed + self._num_solved + seed_offset,
                log=False,
            )
            agent.train(total_timesteps=self.total_timesteps, eval_env=self.env)
            return agent

        agent, discounted_return = solve_best_of_seeds(
            w, self.num_seeds, self.eval_episodes, self.env, train_one
        )
        self._num_solved += self.num_seeds
        self.last_agent = agent  # retained so demos can checkpoint the policy; unused by the study path
        return discounted_return


def make_solver(config, env, seed: int, solver_kwargs: dict):
    """Builds the inner-loop solver declared by `config`, applying any hyperparameter overrides.

    Args:
        config: The environment's EnvironmentConfig (see environments.py).
        env: The (already-seeded) environment instance to solve.
        seed: Base seed for the solver's own reproducibility (e.g. per-policy RL seeding).
        solver_kwargs: Hyperparameter overrides merged on top of config.solver_kwargs.

    Returns:
        An object with a `solve(w) -> np.ndarray` method.
    """
    if config.solver == "exact":
        candidates = np.asarray(env.unwrapped.pareto_front(gamma=config.gamma), dtype=np.float32)
        return ExactEnumerationSolver(candidates)
    elif config.solver == "tabular_q":
        return TabularQLearningSolver(env=env, gamma=config.gamma, seed=seed, **solver_kwargs)
    elif config.solver == "sac":
        return ContinuousSACSolver(env=env, gamma=config.gamma, seed=seed, **solver_kwargs)
    elif config.solver == "discrete_sac":
        return DiscreteSACSolver(env=env, gamma=config.gamma, seed=seed, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver type: {config.solver!r}")

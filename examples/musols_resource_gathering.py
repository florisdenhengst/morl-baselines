"""Benchmarks MUSOLS against OLS on a multi-stakeholder robotics task, with real RL training.

Environment: ResourceGathering (Barrett & Narayanan, 2008, ICML), as provided by mo-gymnasium. A single robot
run (one execution of actions from home to the resources and back) must balance the risk of encountering
enemies against the value of the gold and gem it can bring home -- a single-execution robotics task in the
sense of paper.latex, overseen here by two stakeholders with overlapping but different priorities: a safety
officer who mostly cares about avoiding enemies, and a logistics manager who mostly cares about maximizing
collected resources, while both care about all three objectives to some (differing) degree. With m=2 users and
d=3 objectives (m < d), this is exactly the regime in which MUSOLS is expected to reduce the number of
evaluated corner weights (and thus the number of policies trained) relative to plain OLS.

Unlike the FruitTree benchmark (`musols_fruit_tree.py`), ResourceGathering's dynamics are stochastic and its
state space, while small, is not trivially enumerable, so this benchmark trains a real tabular Q-learning
policy (MOQLearning) per candidate weight, mirroring `examples/ols_dst.py`.
"""

import time

import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.multi_policy.linear_support.musols import MUSOLS
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


GAMMA = 0.95
TIMESTEPS_PER_POLICY = int(3e4)

# Rows are objectives [enemy penalty, gold, gem], columns are stakeholders. Both stakeholders care about all
# three objectives, but the safety officer weighs the enemy penalty far more heavily, while the logistics
# manager weighs resource collection more heavily -- overlapping but not identical, and not one-hot.
STAKEHOLDER_WEIGHTS = np.array(
    [
        [0.70, 0.15],  # -1 if killed by an enemy
        [0.20, 0.55],  # +1 for returning home with gold
        [0.10, 0.30],  # +1 for returning home with the gem
    ],
    dtype=np.float32,
)
STAKEHOLDER_NAMES = ["safety officer", "logistics manager"]


def train_and_evaluate(env, w: np.ndarray) -> np.ndarray:
    """Trains a tabular Q-learning policy scalarized by w and returns its discounted evaluated return."""
    agent = MOQLearning(
        env,
        weights=w,
        learning_rate=0.3,
        gamma=GAMMA,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=int(TIMESTEPS_PER_POLICY * 0.5),
        log=False,
    )
    agent.train(0, total_timesteps=TIMESTEPS_PER_POLICY)
    _, _, _, discounted_return = agent.policy_eval(eval_env=env, weights=w)
    return discounted_return


def run_outer_loop(algo, env, max_seconds: float = 120.0):
    """Runs the OLS/MUSOLS outer loop, training a fresh policy for each candidate weight.

    OLS/MUSOLS are anytime algorithms, so this bounds wall-clock time rather than iteration count -- the same
    approach used in `musols_fruit_tree.py`, kept here for consistency even though a corner-weight blow-up is
    less likely at d=3 than at d=6.

    Returns:
        (algo, converged, elapsed_seconds)
    """
    start = time.perf_counter()
    w = algo.next_weight()
    while not algo.ended() and time.perf_counter() - start < max_seconds:
        value = train_and_evaluate(env, w)
        algo.add_solution(value, w)
        w = algo.next_weight()
    return algo, algo.ended(), time.perf_counter() - start


def main():
    env = mo_gym.make("resource-gathering-v0")
    num_objectives = env.unwrapped.reward_dim

    print("Training OLS (full weight simplex)...")
    ols, ols_converged, ols_time = run_outer_loop(
        LinearSupport(num_objectives=num_objectives, epsilon=0.05, verbose=False), env
    )

    print("Training MUSOLS (restricted to the two stakeholders' weight polytope)...")
    musols, musols_converged, musols_time = run_outer_loop(
        MUSOLS(user_weights=STAKEHOLDER_WEIGHTS, epsilon=0.05, verbose=False), env
    )

    ols_status = "" if ols_converged else " (did NOT converge within the time budget)"
    musols_status = "" if musols_converged else " (did NOT converge within the time budget)"
    print(f"\nOLS    (full CCS):       {len(ols.ccs)} policies, {ols.iteration} trained, {ols_time:.1f}s{ols_status}")
    print(
        f"MUSOLS (restricted CCS): {len(musols.ccs)} policies, {musols.iteration} trained, "
        f"{musols_time:.1f}s{musols_status}"
    )
    print("\nKnown Pareto front (for reference, from the environment's analytical solution):")
    for v in env.unwrapped.pareto_front(gamma=GAMMA):
        print(" ", v)


if __name__ == "__main__":
    main()

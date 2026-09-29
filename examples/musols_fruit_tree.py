"""Benchmarks MUSOLS against OLS on a multi-stakeholder diet-planning problem.

Environment: FruitTree (Yang et al., 2019, https://arxiv.org/abs/1908.08342), as provided by mo-gymnasium.
A single meal plan is chosen by walking root-to-leaf through a binary decision tree; each leaf yields a fixed
6-dimensional nutrient vector (Protein, Carbs, Fats, Vitamins, Minerals, Water). This is a single execution of
actions (one root-to-leaf trajectory) whose outcome must balance the differing nutritional priorities of
multiple stakeholders overseeing the plan -- e.g. an athlete, a dietitian and a physician -- matching the
healthcare/treatment-planning motivation in paper.latex.

Since FruitTree is a small, fully enumerable, deterministic decision tree, the inner-loop optimization used
here is exact (argmax over all leaves) rather than learned by RL: this isolates the outer-loop behaviour of
OLS/MUSOLS (CCS size, corner weights, runtime) from RL training noise, on a real, literature-established set
of multi-objective outcomes (as opposed to synthetic test data).
"""

import time

import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.multi_policy.linear_support.musols import MUSOLS


NUTRIENTS = ["Protein", "Carbs", "Fats", "Vitamins", "Minerals", "Water"]

# Rows are nutrients (in NUTRIENTS order), columns are stakeholders. Each stakeholder's preferences sum to 1
# and put most weight on their priority nutrients, while still caring about the rest -- overlapping but not
# identical, and not one-hot (full disagreement).
STAKEHOLDER_WEIGHTS = np.array(
    [
        [0.35, 0.10, 0.15],  # Protein
        [0.30, 0.10, 0.05],  # Carbs
        [0.05, 0.05, 0.05],  # Fats
        [0.10, 0.30, 0.35],  # Vitamins
        [0.10, 0.30, 0.15],  # Minerals
        [0.10, 0.15, 0.25],  # Water
    ],
    dtype=np.float32,
)
STAKEHOLDER_NAMES = ["athlete", "dietitian", "physician"]


def optimal_leaf(leaves: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact inner-loop solver: the leaf (meal plan) with the highest scalarized nutrient value for weight w."""
    return leaves[np.argmax(leaves @ w)]


def run_outer_loop(algo, leaves: np.ndarray, max_seconds: float = 20.0):
    """Runs the OLS/MUSOLS outer loop, using `optimal_leaf` as the exact inner loop.

    OLS/MUSOLS are anytime algorithms, so this bounds wall-clock time rather than iteration count: later
    iterations get more expensive as the CCS grows (the corner-weight polytope recomputation is combinatorial
    in CCS size, per Theorem 2), so a fixed iteration budget does not bound runtime well. Plain OLS's full
    6-dimensional search failing to converge within a generous time budget on this small, real environment is
    itself a valid, informative result to report -- an extreme but real illustration of that combinatorial
    growth -- not a bug.

    Returns:
        (algo, converged, elapsed_seconds)
    """
    start = time.perf_counter()
    w = algo.next_weight()
    while not algo.ended() and time.perf_counter() - start < max_seconds:
        value = optimal_leaf(leaves, w)
        algo.add_solution(value, w)
        w = algo.next_weight()
    return algo, algo.ended(), time.perf_counter() - start


def main():
    depth = 6
    env = mo_gym.make("fruit-tree-v0", depth=depth).unwrapped
    leaves = np.asarray(env.tree[-(2**depth) :], dtype=np.float32)
    print(f"FruitTree(depth={depth}): {len(leaves)} candidate diets (leaves), {leaves.shape[1]} nutrients.\n")

    ols, ols_converged, ols_time = run_outer_loop(
        LinearSupport(num_objectives=leaves.shape[1], epsilon=0.1, verbose=False), leaves
    )
    musols, musols_converged, musols_time = run_outer_loop(
        MUSOLS(user_weights=STAKEHOLDER_WEIGHTS, epsilon=0.1, verbose=False), leaves
    )

    ols_status = "" if ols_converged else " (did NOT converge within the time budget)"
    musols_status = "" if musols_converged else " (did NOT converge within the time budget)"
    print(f"OLS    (full CCS):       {len(ols.ccs)} diets, {ols.iteration} leaves evaluated, {ols_time:.3f}s{ols_status}")
    print(
        f"MUSOLS (restricted CCS): {len(musols.ccs)} diets, {musols.iteration} leaves evaluated, "
        f"{musols_time:.3f}s{musols_status}"
    )

    print("\nBest diet per stakeholder's own preferences (should each appear in the restricted CCS):")
    restricted = {tuple(np.round(v, 4)) for v in musols.ccs}
    for name, w in zip(STAKEHOLDER_NAMES, STAKEHOLDER_WEIGHTS.T):
        best = optimal_leaf(leaves, w)
        diet = ", ".join(f"{n}={v:.2f}" for n, v in zip(NUTRIENTS, best))
        in_restricted = tuple(np.round(best, 4)) in restricted
        print(f"  {name:>10}: {diet}  (in restricted CCS: {in_restricted})")


if __name__ == "__main__":
    main()

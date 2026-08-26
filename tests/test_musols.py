import time

import numpy as np
import pytest

from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.multi_policy.linear_support.musols import MUSOLS


def run_outer_loop(algo, candidates: np.ndarray, max_iterations: int = 80):
    """Runs the OLS/MUSOLS outer loop against a fixed, known set of candidate payoff vectors.

    Rather than training real policies, the "inner loop" here is exact linear scalarized
    optimization over `candidates`, which is enough to test the outer-loop bookkeeping
    (corner weights, dominance, restriction to the user-induced weight subset) in isolation.
    """
    w = algo.next_weight()
    iterations = 0
    while not algo.ended() and iterations < max_iterations:
        value = candidates[np.argmax(candidates @ w)]
        algo.add_solution(value, w)
        w = algo.next_weight()
        iterations += 1
    assert iterations < max_iterations, "Outer loop did not converge within max_iterations."
    return algo


def make_candidates(num_objectives: int, num_points: int, seed: int) -> np.ndarray:
    """Samples a small set of candidate payoff vectors, some dominated and some not."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_points, num_objectives)).astype(np.float32)


def make_user_weights(num_objectives: int, num_users: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(num_objectives), size=num_users).T.astype(np.float32)


def brute_force_restricted_ccs(candidates: np.ndarray, W: np.ndarray, num_samples: int = 4000, seed: int = 0) -> set:
    """Ground truth restricted CCS: densely samples consensus weights alpha and records, for each, which
    candidate is scalarized-optimal for w = W @ alpha. With enough samples this recovers every payoff vector
    that is optimal for some weight in Omega_W, independent of MUSOLS's own corner-weight search procedure.
    """
    rng = np.random.default_rng(seed)
    alphas = rng.dirichlet(np.ones(W.shape[1]), size=num_samples)
    w = alphas @ W.T
    best_idx = np.argmax(w @ candidates.T, axis=1)
    return {tuple(np.round(candidates[i], 4)) for i in np.unique(best_idx)}


@pytest.mark.parametrize(
    "num_objectives,num_users,seed",
    [(2, 2, 0), (3, 2, 1), (3, 3, 2), (4, 2, 3), (4, 3, 4)],
)
def test_matches_brute_force_restricted_ccs(num_objectives, num_users, seed):
    """MUSOLS must recover exactly the restricted CCS obtained by densely sampling the consensus weight simplex."""
    candidates = make_candidates(num_objectives, num_points=10, seed=seed)
    W = make_user_weights(num_objectives, num_users, seed=seed)

    musols = run_outer_loop(MUSOLS(user_weights=W, epsilon=0.01, verbose=False), candidates)
    restricted_ccs = {tuple(np.round(v, 4)) for v in musols.ccs}

    assert restricted_ccs == brute_force_restricted_ccs(candidates, W, seed=seed)


@pytest.mark.parametrize(
    "num_objectives,num_users,seed",
    [(2, 2, 0), (3, 2, 1), (4, 3, 4)],
)
def test_weight_support_is_consistent_with_user_weights(num_objectives, num_users, seed):
    """Every weight in the restricted CCS's weight support must equal W @ alpha for its consensus weight alpha."""
    candidates = make_candidates(num_objectives, num_points=10, seed=seed)
    W = make_user_weights(num_objectives, num_users, seed=seed)

    musols = run_outer_loop(MUSOLS(user_weights=W, epsilon=0.01, verbose=False), candidates)

    for w, alpha in zip(musols.get_weight_support(), musols.get_consensus_weight_support()):
        np.testing.assert_allclose(W @ alpha, w, atol=1e-4)


@pytest.mark.parametrize("num_objectives,seed", [(2, 0), (3, 1), (4, 2)])
def test_matches_ols_under_extreme_disagreement(num_objectives, seed):
    """If every objective is the sole priority of exactly one user (W is the identity matrix), the reachable
    consensus weight polytope Omega_W is the full weight simplex, so MUSOLS must recover the same CCS as OLS.
    """
    candidates = make_candidates(num_objectives, num_points=10, seed=seed)
    W = np.eye(num_objectives, dtype=np.float32)

    ols = run_outer_loop(LinearSupport(num_objectives=num_objectives, epsilon=0.01, verbose=False), candidates)
    musols = run_outer_loop(MUSOLS(user_weights=W, epsilon=0.01, verbose=False), candidates)

    full_ccs = {tuple(np.round(v, 4)) for v in ols.ccs}
    restricted_ccs = {tuple(np.round(v, 4)) for v in musols.ccs}
    assert full_ccs == restricted_ccs


def test_single_user_reduces_to_a_single_point():
    """With a single user (m=1), Omega_W is the single point w_1, so the restricted CCS has one payoff."""
    num_objectives, seed = 3, 11
    candidates = make_candidates(num_objectives, num_points=10, seed=seed)
    w1 = make_user_weights(num_objectives, num_users=1, seed=seed)

    musols = run_outer_loop(MUSOLS(user_weights=w1, epsilon=0.01, verbose=False), candidates)

    assert len(musols.ccs) == 1
    expected = candidates[np.argmax(candidates @ w1[:, 0])]
    np.testing.assert_allclose(musols.ccs[0], expected, atol=1e-4)


def test_add_solution_rejects_stale_weight():
    W = np.eye(3, dtype=np.float32)
    musols = MUSOLS(user_weights=W, verbose=False)
    musols.next_weight()
    stale_w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        musols.add_solution(np.zeros(3, dtype=np.float32), stale_w)


def test_musols_performance_smoke_test():
    """With few users and many objectives (m << d), MUSOLS restricts the search to a low-dimensional consensus
    weight polytope. Theorem 2 predicts this reduces the number of evaluated corner weights (and thus runtime)
    relative to plain OLS searching the full objective weight simplex; this checks that empirically.
    """
    num_objectives, num_users, seed = 6, 2, 0
    candidates = make_candidates(num_objectives, num_points=8, seed=seed)
    W = make_user_weights(num_objectives, num_users, seed=seed)

    start = time.perf_counter()
    ols = run_outer_loop(LinearSupport(num_objectives=num_objectives, epsilon=0.01, verbose=False), candidates)
    ols_time = time.perf_counter() - start

    start = time.perf_counter()
    musols = run_outer_loop(MUSOLS(user_weights=W, epsilon=0.01, verbose=False), candidates)
    musols_time = time.perf_counter() - start

    print("OLS took %d iterations, MUSOLS took %d iterations" % (ols.iteration, musols.iteration))
    assert musols.iteration <= ols.iteration
    print("OLS computed %d corner weights, MUSOLS computed %d corner weights" % (len(ols.compute_corner_weights()), len(musols.compute_corner_weights())))
    assert len(musols.compute_corner_weights()) <= len(ols.compute_corner_weights())
    print("Timing results: OLS took %.4f s, MUSOLS took %.4f s" % (ols_time, musols_time))
    assert musols_time < ols_time

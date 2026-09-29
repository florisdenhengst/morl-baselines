"""Multi-User Subset Optimistic Linear Support (MUSOLS) implementation."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


logger = logging.getLogger(__name__)


def prune_redundant_users(user_weights: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Drops user weight vectors that lie in the convex hull of the other users' weight vectors.

    Omega_W is the convex hull of W's columns, so a column that is already a convex combination of the others
    contributes nothing to it: removing such columns leaves Omega_W *exactly* unchanged while shrinking the
    consensus space MUSOLS searches. This matters most when there are more users than objectives (m > d),
    where the columns are necessarily affinely dependent: e.g. for d=2 any number of users spans at most a line
    segment, whose hull needs only its 2 endpoints, so an m=10 panel is searched as if m=2.

    Columns are tested in order against the columns retained so far plus the ones not yet tested, so exact
    duplicates keep their first occurrence rather than eliminating each other.

    Args:
        user_weights (np.ndarray): User preference matrix W of shape (num_objectives, num_users).

    Returns:
        Tuple of the reduced matrix (num_objectives, num_retained) and the retained column indices, in the
        original column order.
    """
    num_users = user_weights.shape[1]
    if num_users <= 1:
        return user_weights, list(range(num_users))

    retained = list(range(num_users))
    for index in range(num_users):
        others = [j for j in retained if j != index]
        if not others:
            continue
        # Feasibility LP: is w_index a convex combination of the other retained columns?
        hull = user_weights[:, others]
        a_eq = np.vstack([hull, np.ones(len(others))])
        b_eq = np.concatenate([user_weights[:, index], [1.0]])
        solution = linprog(c=np.zeros(len(others)), A_eq=a_eq, b_eq=b_eq, bounds=(0, None))
        if solution.success:
            retained.remove(index)
    return user_weights[:, retained], retained


class MUSOLS:
    """Multi-User Subset Optimistic Linear Support for computing a restricted CCS given known user preferences.

    Extends Optimistic Linear Support (OLS, see `LinearSupport`) to a multi-user setting in which a latent
    consensus utility u*(v) = (W @ alpha)^T v is only known to be *some* convex combination, with unknown weights
    alpha in the (m-1)-simplex, of m known user utilities with weight vectors w_1, ..., w_m (the columns of the
    user preference matrix W, of shape (num_objectives, num_users)). Rather than searching the full objective
    weight simplex, MUSOLS restricts the search to the reachable consensus weight polytope
    Omega_W = {W @ alpha | alpha in the (m-1)-simplex}, i.e. the convex hull of the columns of W.

    Since for any alpha, alpha^T (W^T v) = (W @ alpha)^T v, searching Omega_W is equivalent to running standard
    OLS in the m-dimensional space of consensus weights alpha, using the projected value vectors W^T v. MUSOLS
    therefore delegates corner weight computation, dominance checks and prioritization to an internal
    `LinearSupport` instance of dimensionality m (instead of num_objectives), and only keeps track of the
    associated payoff vectors in the original objective space.

    This gives the same guarantees as OLS (Section 3.3 of http://roijers.info/pub/thesis.pdf), but restricted to
    the multi-user subset CCS instead of the full CCS.
    """

    def __init__(
        self,
        user_weights: np.ndarray,
        epsilon: float = 0.0,
        verbose: bool = True,
        prune_users: bool = True,
    ):
        """Initialize MUSOLS.

        Args:
            user_weights (np.ndarray): User preference matrix W of shape (num_objectives, num_users). Each
                column w_i is the known weight vector of user i in the objective weight simplex.
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to True.
            prune_users (bool): Whether to drop users whose weight vector lies in the convex hull of the
                others before searching. This leaves Omega_W exactly unchanged (see `prune_redundant_users`)
                while shrinking the searched consensus space, and is what keeps the search dimension from
                growing with the number of users once they become affinely dependent (m > d). Disable only to
                measure its effect. Defaults to True.
        """
        self.W = np.asarray(user_weights, dtype=np.float32)
        assert self.W.ndim == 2, "user_weights must be a (num_objectives, num_users) matrix."
        self.num_objectives, self.num_users = self.W.shape
        self.epsilon = epsilon
        self.verbose = verbose

        # Searching the hull of a set of points only needs that set's vertices, so redundant users are dropped
        # up front. `self.W` stays the caller's original matrix: reported consensus weights are padded back
        # into it (with zero weight on pruned users), so `W @ alpha == w` still holds for the caller.
        if prune_users:
            self._search_W, self.retained_users = prune_redundant_users(self.W)
        else:
            self._search_W, self.retained_users = self.W, list(range(self.num_users))
        self._num_search_users = self._search_W.shape[1]
        if self.verbose and self._num_search_users < self.num_users:
            logger.info(
                "Pruned %d redundant user(s); searching a %d-dimensional consensus space instead of %d.",
                self.num_users - self._num_search_users,
                self._num_search_users,
                self.num_users,
            )

        self.ccs: List[np.ndarray] = []  # Payoff vectors, aligned with self._ls.ccs.

        # Delegate corner weight computation, dominance checks and prioritization to a standard LinearSupport
        # instance operating in the consensus weight (alpha) space of the retained users.
        self._ls = LinearSupport(num_objectives=self._num_search_users, epsilon=epsilon, verbose=False)
        self._pending_alpha: Optional[np.ndarray] = None
        self._pending_w: Optional[np.ndarray] = None

    def _expand_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """Pads a consensus weight over the retained users back to one over all of the caller's users."""
        if self._num_search_users == self.num_users:
            return alpha
        expanded = np.zeros(self.num_users, dtype=np.float32)
        expanded[self.retained_users] = alpha
        return expanded

    @property
    def iteration(self) -> int:
        """Number of solutions added so far (mirrors `LinearSupport.iteration`)."""
        return self._ls.iteration

    def next_weight(self) -> Optional[np.ndarray]:
        """Returns the next objective weight vector w = W @ alpha with the highest priority, or None if done.

        Returns:
            np.ndarray: Next objective weight vector, to be used by the inner-loop single-objective algorithm.
                None if there are no more candidates to try.
        """
        alpha = self._ls.next_weight(algo="ols")
        if alpha is None:
            if self.verbose:
                logger.info("There are no corner weights in the queue. Returning None.")
            self._pending_alpha, self._pending_w = None, None
            return None

        w = self._search_W @ alpha
        self._pending_alpha, self._pending_w = alpha, w
        if self.verbose:
            logger.info("Next consensus weight: %s -> objective weight: %s", alpha, w)
        return w

    def ended(self) -> bool:
        """Returns True if there are no more consensus weight vectors to test.

        Warning: This method must be called AFTER calling next_weight().
        Ex: w = musols.next_weight()
            if musols.ended():
                print("MUSOLS ended.")
        """
        return self._ls.ended()

    def add_solution(self, value: np.ndarray, w: np.ndarray) -> List[int]:
        """Add a new value vector, optimal for the objective weight w returned by the last call to next_weight().

        Args:
            value (np.ndarray): New value vector.
            w (np.ndarray): The objective weight vector returned by the most recent call to next_weight().

        Returns:
            List of indices of value vectors removed from the CCS for being dominated.
        """
        if self._pending_alpha is None or not np.allclose(w, self._pending_w):
            raise ValueError("add_solution() must be called with the weight vector returned by next_weight().")
        alpha = self._pending_alpha
        self._pending_alpha, self._pending_w = None, None

        if self.verbose:
            logger.info("Adding value=%s for consensus weight=%s (w=%s) to the restricted CCS.", value, alpha, w)

        # LinearSupport.add_solution() returns the sentinel [len(ccs)] (an out-of-range index) when the
        # candidate is dominated and discarded; removed indices from actual dominated-value removals are
        # always < len(ccs), so this check unambiguously distinguishes the two cases.
        n_before = len(self._ls.ccs)
        removed_indx = self._ls.add_solution(self._search_W.T @ value, alpha)
        if removed_indx == [n_before]:
            return removed_indx

        for i in sorted(removed_indx, reverse=True):
            self.ccs.pop(i)
        self.ccs.append(value)
        return removed_indx

    def get_weight_support(self) -> List[np.ndarray]:
        """Returns the objective weight support {W @ alpha | alpha in consensus weight support} of the CCS.

        Returns:
            List[np.ndarray]: List of objective weight vectors, one per value vector in the restricted CCS.
        """
        return [self._search_W @ alpha for alpha in self._ls.get_weight_support()]

    def get_consensus_weight_support(self) -> List[np.ndarray]:
        """Returns the consensus weights alpha associated with the restricted CCS.

        Each alpha is expressed over *all* of the caller's users, with zero weight on any pruned user, so
        `W @ alpha` reproduces the corresponding objective weight regardless of pruning.

        Returns:
            List[np.ndarray]: List of consensus weight vectors, one per value vector in the restricted CCS.
        """
        return [self._expand_alpha(alpha) for alpha in self._ls.get_weight_support()]

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        """Returns the objective-space corner weights of the current restricted CCS.

        Args:
            top_k: If not None, returns the top_k corner weights.

        Returns:
            List[np.ndarray]: List of objective weight vectors W @ alpha_c for the current corner weights alpha_c.
        """
        return [self._search_W @ alpha for alpha in self._ls.get_corner_weights(top_k=top_k)]

    def compute_corner_weights(self) -> List[np.ndarray]:
        """Returns the objective-space corner weights for the current restricted CCS.

        Unlike `get_corner_weights()`, which returns the (epsilon- and visited-weight-filtered) queue, this
        recomputes the full set of corner weights of the polytope induced by the current restricted CCS. See
        `LinearSupport.compute_corner_weights()`.

        Returns:
            List[np.ndarray]: List of objective weight vectors W @ alpha_c for every corner weight alpha_c.
        """
        return [self._search_W @ alpha for alpha in self._ls.compute_corner_weights()]


if __name__ == "__main__":

    def _solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    num_objectives = 3
    # Two users with known, distinct weight vectors over the 3 objectives.
    W = np.array([[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]], dtype=np.float32)
    musols = MUSOLS(user_weights=W, epsilon=0.0001, verbose=True)
    w = musols.next_weight()
    while not musols.ended():
        print("w:", w)
        value = _solve(w)
        musols.add_solution(value, w)
        w = musols.next_weight()

"""Multi-User Subset Optimistic Linear Support (MUSOLS) implementation."""

import logging
from typing import List, Optional

import numpy as np

from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


logger = logging.getLogger(__name__)


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
    ):
        """Initialize MUSOLS.

        Args:
            user_weights (np.ndarray): User preference matrix W of shape (num_objectives, num_users). Each
                column w_i is the known weight vector of user i in the objective weight simplex.
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to True.
        """
        self.W = np.asarray(user_weights, dtype=np.float32)
        assert self.W.ndim == 2, "user_weights must be a (num_objectives, num_users) matrix."
        self.num_objectives, self.num_users = self.W.shape
        self.epsilon = epsilon
        self.verbose = verbose

        self.ccs: List[np.ndarray] = []  # Payoff vectors, aligned with self._ls.ccs.

        # Delegate corner weight computation, dominance checks and prioritization to a standard LinearSupport
        # instance operating in the num_users-dimensional consensus weight (alpha) space.
        self._ls = LinearSupport(num_objectives=self.num_users, epsilon=epsilon, verbose=False)
        self._pending_alpha: Optional[np.ndarray] = None
        self._pending_w: Optional[np.ndarray] = None

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

        w = self.W @ alpha
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
        removed_indx = self._ls.add_solution(self.W.T @ value, alpha)
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
        return [self.W @ alpha for alpha in self._ls.get_weight_support()]

    def get_consensus_weight_support(self) -> List[np.ndarray]:
        """Returns the consensus weights alpha associated with the restricted CCS.

        Returns:
            List[np.ndarray]: List of consensus weight vectors, one per value vector in the restricted CCS.
        """
        return self._ls.get_weight_support()

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        """Returns the objective-space corner weights of the current restricted CCS.

        Args:
            top_k: If not None, returns the top_k corner weights.

        Returns:
            List[np.ndarray]: List of objective weight vectors W @ alpha_c for the current corner weights alpha_c.
        """
        return [self.W @ alpha for alpha in self._ls.get_corner_weights(top_k=top_k)]

    def compute_corner_weights(self) -> List[np.ndarray]:
        """Returns the objective-space corner weights for the current restricted CCS.

        Unlike `get_corner_weights()`, which returns the (epsilon- and visited-weight-filtered) queue, this
        recomputes the full set of corner weights of the polytope induced by the current restricted CCS. See
        `LinearSupport.compute_corner_weights()`.

        Returns:
            List[np.ndarray]: List of objective weight vectors W @ alpha_c for every corner weight alpha_c.
        """
        return [self.W @ alpha for alpha in self._ls.compute_corner_weights()]


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

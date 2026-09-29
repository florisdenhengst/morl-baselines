"""Baseline outer-loop algorithms for the MUSOLS-vs-OLS comparison.

Plain OLS is not a fair baseline for isolating the value of MUSOLS's *search strategy*: it does not use the
stakeholder weight matrix W at all, so any MUSOLS-vs-OLS gap conflates two separate effects -- (1) restricting
the search to the induced consensus weight polytope Omega_W, and (2) MUSOLS's optimistic corner-weight-guided
exploration of that polytope. `RandomOmegaWSearch` isolates effect (2): it is given the exact same information
as MUSOLS (the weight matrix W, hence the same restriction to Omega_W) but selects candidate weights by
uniform random sampling over the consensus simplex instead of MUSOLS's optimistic corner-weight search. Any
remaining MUSOLS-vs-RandomOmegaWSearch gap, given the same compute budget, is attributable to the search
strategy itself, not to having privileged information the baseline lacks. This mirrors how OLS itself has
historically been benchmarked against naive uniform weight sampling (Roijers, 2016, PhD thesis, Section 3.3).
"""

from typing import Optional

import numpy as np

from morl_baselines.multi_policy.linear_support.musols import MUSOLS


class RandomOmegaWSearch(MUSOLS):
    """MUSOLS with its optimistic corner-weight search replaced by uniform random sampling over Omega_W.

    An anytime baseline: `ended()` always returns False, so it only stops when the outer-loop harness's
    wall-clock budget runs out (see `outer_loop.run_outer_loop`). Reuses MUSOLS's dominance bookkeeping
    (`add_solution`, `get_weight_support`, `iteration`, ...) unchanged via inheritance -- only weight selection
    differs.
    """

    def __init__(self, user_weights: np.ndarray, seed: int = 0, verbose: bool = False):
        """Initialize the baseline.

        Args:
            user_weights (np.ndarray): Same user preference matrix W as passed to MUSOLS.
            seed (int): Seed for the uniform weight sampler, for reproducibility.
            verbose (bool): Defaults to False.
        """
        super().__init__(user_weights=user_weights, epsilon=0.0, verbose=verbose)
        self.rng = np.random.default_rng(seed)

    def next_weight(self) -> np.ndarray:
        """Returns a uniformly random objective weight vector w = W @ alpha, alpha ~ Dirichlet(1, ..., 1).

        Sampling is over the users MUSOLS retained after pruning, which span exactly the same Omega_W (see
        `prune_redundant_users`). Sampling over redundant users instead would waste draws on duplicate
        parameterizations of the same consensus weight, handicapping the baseline for no good reason.
        """
        alpha = self.rng.dirichlet(np.ones(self._num_search_users)).astype(np.float32)
        w = self._search_W @ alpha
        self._pending_alpha, self._pending_w = alpha, w
        return w

    def ended(self) -> bool:
        """Always False: this is an anytime baseline with no natural stopping point."""
        return False


class VertexOnlySearch(MUSOLS):
    """Solves for each stakeholder's own preference and stops: the naive multi-user baseline.

    This is what a practitioner does without any dedicated algorithm -- hand every stakeholder the policy that
    is optimal for *their* weight vector and present the resulting m policies as the trade-off set. It
    evaluates exactly the m vertices of Omega_W and never explores its interior, so comparing against it
    isolates what is gained by covering the *consensus* space rather than only the individual optima. Any
    consensus alpha placing weight on several stakeholders at once can be served strictly better by an
    interior point, so the gap here is precisely the value of treating consensus as its own search problem.

    Reuses MUSOLS's dominance bookkeeping unchanged, so its returned set is filtered identically: a
    stakeholder's own optimum is dropped if another stakeholder's optimum dominates it everywhere in Omega_W.
    """

    def __init__(self, user_weights: np.ndarray, verbose: bool = False):
        """Initialize the baseline.

        Args:
            user_weights (np.ndarray): Same user preference matrix W as passed to MUSOLS.
            verbose (bool): Defaults to False.
        """
        super().__init__(user_weights=user_weights, epsilon=0.0, verbose=verbose)
        self._next_vertex = 0

    def next_weight(self) -> Optional[np.ndarray]:
        """Returns the next stakeholder's own weight vector, or None once every stakeholder has been served."""
        if self._next_vertex >= self._num_search_users:
            self._pending_alpha, self._pending_w = None, None
            return None
        alpha = np.zeros(self._num_search_users, dtype=np.float32)
        alpha[self._next_vertex] = 1.0
        self._next_vertex += 1
        w = self._search_W @ alpha
        self._pending_alpha, self._pending_w = alpha, w
        return w

    def ended(self) -> bool:
        """Returns True once every stakeholder's own optimum has been solved for."""
        return self._next_vertex >= self._num_search_users

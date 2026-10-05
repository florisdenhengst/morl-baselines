"""Linear Support implementation."""

import random
from copy import deepcopy
from typing import List, Optional

import cvxpy as cp
from fractions import Fraction

import numpy as np
from cvxpy import SolverError
from gymnasium.core import Env

from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.weights import extrema_weights


try:
    import cdd
except ImportError as e:
    raise ImportError(
        "To use this feature, you need to install the optional dependency pycddlib==2.1.6 as stated in pyproject.toml of morl-baselines"
    ) from e


np.set_printoptions(precision=4)


class LinearSupport:
    """Linear Support for computing corner weights when using linear utility functions.

    Implements both

    Optimistic Linear Support (OLS) algorithm:
    Paper: (Section 3.3 of http://roijers.info/pub/thesis.pdf).

    Generalized Policy Improvement Linear Support (GPI-LS) algorithm:
    Paper: https://arxiv.org/abs/2301.07784
    """

    def __init__(
        self,
        num_objectives: int,
        epsilon: float = 0.0,
        verbose: bool = True,
        monotone: bool = False,
        reopen_delta: Optional[float] = None,
        max_resolves: int = 2,
        optimism: float = 0.0,
    ):
        """Initialize Linear Support.

        Args:
            num_objectives (int): Number of objectives
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to False.
            monotone (bool): If True, a solve whose value is beaten at its own weight by a vector already
                in the CCS is treated as a failed solve and discarded. See `add_solution`. Defaults to
                False (the classical behaviour).
            reopen_delta (float, optional): If set, a solved weight may be tried again once another value
                vector is found that beats the one its own solve produced, by more than this margin. This
                relaxes OLS's assumption of an exact inner solver; see `_is_closed`. None (the default)
                keeps the classical behaviour of never revisiting a weight.
            max_resolves (int): Cap on how many times any single weight may be reopened. Defaults to 2.
            optimism (float): Relative slack added to every recorded value when computing the optimistic
                upper bound, acknowledging that an inexact solver may have *underestimated* it. Without this
                the bound at a visited weight is capped by that weight's own recorded value, so the search
                is structurally incapable of suspecting its own solves. See `max_value_lp`. 0.0 (the default)
                is classical OLS.
        """
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.visited_weights = []  # List of already tested weight vectors
        self.ccs = []
        self.weight_support = []  # List of weight vectors for each value vector in the CCS
        self.queue = []
        self.iteration = 0
        self.ols_ended = False
        self.verbose = verbose
        self.monotone = monotone
        self.reopen_delta = reopen_delta
        self.max_resolves = max_resolves
        self.optimism = optimism
        # Scalarized value the *solver itself* returned at each visited weight, as distinct from the
        # envelope there (the best any known vector achieves). Classical OLS conflates the two, because with
        # an exact solver they are equal by definition. They come apart exactly when a solve underperforms,
        # which is the signal `_is_closed` acts on.
        self._solver_value = {}
        self._resolves = {}
        for w in extrema_weights(self.num_objectives):
            self.queue.append((float("inf"), w))

    @staticmethod
    def _weight_key(w: np.ndarray):
        """Hashable key for a weight vector, rounded so float noise does not create spurious new entries."""
        return tuple(np.round(np.asarray(w, dtype=float), 6))

    def _record_solve(self, value: np.ndarray, w: np.ndarray) -> None:
        """Records the *raw* value the solver achieved at `w`, for the reopening test in `_is_closed`.

        Deliberately not floored at the envelope, even under `monotone`: the gap between what the solver
        achieved here and what is achievable here is precisely the certificate `_is_closed` needs, so
        flooring it would silence the signal the two options are meant to compose on. A reopened weight is
        credited with its best attempt rather than its latest, so a second bad draw cannot reopen it again
        on the strength of being worse than the first.
        """
        key = self._weight_key(w)
        achieved = float(np.dot(np.asarray(value, dtype=float), np.asarray(w, dtype=float)))
        self._solver_value[key] = max(self._solver_value.get(key, -np.inf), achieved)

    def _is_closed(self, w: np.ndarray) -> bool:
        """Whether `w` has been solved and should not be tried again.

        Classical OLS closes a weight permanently the moment it has been solved once. That is sound only for
        an exact inner solver: the inference "I solved w and found nothing better, therefore nothing better
        exists at w" fails as soon as `solve` can return a suboptimal policy, and nothing in the algorithm
        can ever undo it. One unlucky training run therefore abandons a region of weight space for good.

        With `reopen_delta` set, a solved weight is reopened when the run has produced a *certificate* that
        its solve was suboptimal: some vector already in the CCS beats what the solver achieved at w, by more
        than the margin. The certificate is sound without knowing the true optimum -- the competing vector is
        achievable and scores higher at w, so the solve demonstrably was not the argmax. It is also exactly
        the information classical OLS already has and throws away.

        Termination is preserved: each reopening requires a strict improvement of at least `reopen_delta` at
        that weight, values are bounded, and `max_resolves` caps the count regardless. With an exact solver
        no certificate can ever fire, so this reduces to the classical rule at identical cost.
        """
        if not any(np.allclose(w, wv) for wv in self.visited_weights):
            return False
        # Either mechanism alone enables revisiting; neither means classical OLS.
        if self.reopen_delta is None and self.optimism <= 0.0:
            return True
        key = self._weight_key(w)
        if self._resolves.get(key, 0) >= self.max_resolves:
            return True
        # Speculative reopening. The certificate below is sound but weak: it can only fire once a *better*
        # policy has turned up at some other weight, which is circular precisely when the search is short
        # enough to be truncated by one bad solve. With `optimism` in play the weight stays eligible on
        # suspicion rather than proof, and `max_resolves` bounds the cost.
        if self.optimism > 0.0:
            return False
        if self.reopen_delta is None:
            return True
        achieved = self._solver_value.get(key)
        envelope = self.max_scalarized_value(w)
        if achieved is None or envelope is None:
            return True
        return not (envelope > achieved + self.reopen_delta)

    def next_weight(
        self, algo: str = "ols", gpi_agent: Optional[MOPolicy] = None, env: Optional[Env] = None, rep_eval: int = 1
    ) -> np.ndarray:
        """Returns the next weight vector with highest priority.

        Args:
            algo (str): Algorithm to use. Either 'ols' or 'gpi-ls'.
            gpi_agent (Optional[MOPolicy]): Agent to use for GPI-LS.
            env (Optional[Env]): Environment to use for GPI-LS.
            rep_eval (int): Number of times to evaluate the agent in GPI-LS.

        Returns:
            np.ndarray: Next weight vector
        """
        if len(self.ccs) > 0:
            W_corner = self.compute_corner_weights()
            if self.verbose:
                print("W_corner:", W_corner, "W_corner size:", len(W_corner))

            self.queue = []
            for wc in W_corner:
                if algo == "ols":
                    priority = self.ols_priority(wc)

                elif algo == "gpi-ls":
                    if gpi_agent is None:
                        raise ValueError("GPI-LS requires passing a GPI agent.")
                    gpi_expanded_set = [policy_evaluation_mo(gpi_agent, env, wc, rep=rep_eval)[3] for wc in W_corner]
                    priority = self.gpi_ls_priority(wc, gpi_expanded_set)

                if self.epsilon is None or priority >= self.epsilon:
                    # OLS does not try the same weight vector twice -- unless `reopen_delta` is set and the
                    # run has certified that this weight's own solve was suboptimal (see `_is_closed`).
                    if not (algo == "ols" and self._is_closed(wc)):
                        self.queue.append((priority, wc))

            if len(self.queue) > 0:
                # Sort in descending order of priority
                self.queue.sort(key=lambda t: t[0], reverse=True)
                # If all priorities are 0, shuffle the queue to avoid repeating weights every iteration
                if self.queue[0][0] == 0.0:
                    random.shuffle(self.queue)

        if self.verbose:
            print("CCS:", self.ccs, "CCS size:", len(self.ccs))

        if len(self.queue) == 0:
            if self.verbose:
                print("There are no corner weights in the queue. Returning None.")
            self.ols_ended = True
            return None
        else:
            next_w = self.queue.pop(0)[1]
            if self.verbose:
                print("Next weight:", next_w)
            return next_w

    def get_weight_support(self) -> List[np.ndarray]:
        """Returns the weight support of the CCS.

        Returns:
            List[np.ndarray]: List of weight vectors of the CCS

        """
        return deepcopy(self.weight_support)

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        """Returns the corner weights of the current CCS.

        Args:
            top_k: If not None, returns the top_k corner weights.

        Returns:
            List[np.ndarray]: List of corner weights.
        """
        weights = [w.copy() for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def ended(self) -> bool:
        """Returns True if there are no more corner weights to test.

        Warning: This method must be called AFTER calling next_weight().
        Ex: w = ols.next_weight()
            if ols.ended():
                print("OLS ended.")
        """
        return self.ols_ended

    def add_solution(self, value: np.ndarray, w: np.ndarray) -> List[int]:
        """Add new value vector optimal to weight w.

        Args:
            value (np.ndarray): New value vector
            w (np.ndarray): Weight vector

        Returns:
            List of indices of value vectors removed from the CCS for being dominated.
        """
        if self.verbose:
            print(f"Adding value={value} for weight={w} to CCS.")

        self.iteration += 1
        key = self._weight_key(w)
        if any(np.allclose(w, wv) for wv in self.visited_weights):
            self._resolves[key] = self._resolves.get(key, 0) + 1
        self.visited_weights.append(w)
        self._record_solve(value, w)

        if self.is_dominated(value):
            if self.verbose:
                print(f"Value {value} is dominated. Discarding.")
            return [len(self.ccs)]

        # A solve that loses at its own weight to a vector already found is not an argmax, so under an exact
        # solver it cannot occur: a non-dominated vector is by definition optimal somewhere, and a solve at w
        # returns the optimum at w. It occurs only when the inner solver underperforms, and admitting the
        # result inflates the coverage set with a policy no weight actually selects. Dropping it keeps the
        # returned set to policies some consensus weight would really choose.
        if self.monotone:
            envelope = self.max_scalarized_value(w)
            achieved = float(np.dot(np.asarray(value, dtype=float), np.asarray(w, dtype=float)))
            if envelope is not None and float(envelope) > achieved:
                if self.verbose:
                    print(f"Value {value} is beaten at its own weight {w}. Discarding as a failed solve.")
                return [len(self.ccs)]

        removed_indx = self.remove_obsolete_values(value)

        self.ccs.append(value)
        self.weight_support.append(w)

        return removed_indx

    def ols_priority(self, w: np.ndarray) -> float:
        """Get the priority of a weight vector for OLS.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """
        max_value_ccs = self.max_scalarized_value(w)
        max_optimistic_value = self.max_value_lp(w)
        priority = max_optimistic_value - max_value_ccs
        return priority

    def gpi_ls_priority(self, w: np.ndarray, gpi_expanded_set: List[np.ndarray]) -> float:
        """Get the priority of a weight vector for GPI-LS.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """

        def best_vector(values, w):
            max_v = values[0]
            for i in range(1, len(values)):
                if values[i] @ w > max_v @ w:
                    max_v = values[i]
            return max_v

        max_value_ccs = self.max_scalarized_value(w)
        max_value_gpi = best_vector(gpi_expanded_set, w)
        max_value_gpi = np.dot(max_value_gpi, w)
        priority = max_value_gpi - max_value_ccs

        return priority

    def max_scalarized_value(self, w: np.ndarray) -> Optional[float]:
        """Returns the maximum scalarized value for weight vector w.

        Args:
            w: Weight vector

        Returns:
            Maximum scalarized value for weight vector w.
        """
        if len(self.ccs) == 0:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        """Removes the values vectors which are no longer optimal for any weight vector after adding the new value vector.

        Args:
            value (np.ndarray): New value vector

        Returns:
            The indices of the removed values.
        """
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            weights_optimal = [
                w
                for w in self.visited_weights
                if np.dot(self.ccs[i], w) == self.max_scalarized_value(w) and np.dot(value, w) < np.dot(self.ccs[i], w)
            ]
            if len(weights_optimal) == 0:
                if self.verbose:
                    print("removed value", self.ccs[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                self.weight_support.pop(i)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        """Returns an upper-bound for the maximum value of the scalarized objective.

        Args:
            w_new: New weight vector

        Returns:
            Upper-bound for the maximum value of the scalarized objective.
        """
        # No upper bound if no values in CCS
        if len(self.ccs) == 0:
            return float("inf")

        w = cp.Parameter(self.num_objectives)
        w.value = w_new
        v = cp.Variable(self.num_objectives)

        W_ = np.vstack(self.visited_weights)
        W = cp.Parameter(W_.shape)
        W.value = W_

        V_ = np.array([self.max_scalarized_value(weight) for weight in self.visited_weights])
        # Each recorded value is a *lower* bound on what is achievable at its weight whenever the inner
        # solver is inexact, so treating it as an equality (optimism = 0) propagates one bad solve into a
        # globally over-tight bound and suppresses priorities everywhere, not just at the weight that failed.
        if self.optimism > 0.0:
            V_ = V_ + self.optimism * np.maximum(np.abs(V_), 1e-12)
        V = cp.Parameter(V_.shape)
        V.value = V_

        # Maximum value for weight vector w
        objective = cp.Maximize(w @ v)
        # such that it is consistent with other optimal values for other visited weights
        constraints = [W @ v <= V]
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(verbose=False)
        except SolverError:
            print("ECOS solver error, trying another one.")
            result = prob.solve(solver=cp.SCS, verbose=False)
        return result

    def compute_corner_weights(self) -> List[np.ndarray]:
        """Returns the corner weights for the current set of values.

        See http://roijers.info/pub/thesis.pdf Definition 19.
        Obs: there is a typo in the definition of the corner weights in the thesis, the >= sign should be <=.

        Returns:
            List of corner weights.
        """
        A = np.vstack(self.ccs)
        A = np.round(A, decimals=4)  # Round to avoid numerical issues
        A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)

        A_plus = np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)
        A_plus = -np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)

        for i in range(self.num_objectives):
            A_plus = np.zeros(A.shape[1]).reshape(1, -1)
            A_plus[0, i] = -1
            A = np.concatenate((A, A_plus), axis=0)

        b = np.zeros(len(self.ccs) + 2 + self.num_objectives)
        b[len(self.ccs)] = 1
        b[len(self.ccs) + 1] = -1

        def _vertices_from(rows, number_type):
            mat = cdd.Matrix(rows, number_type=number_type)
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g, dtype=float)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1:
                    continue
                if i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        def compute_poly_vertices(A, b):
            # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
            b = b.reshape((b.shape[0], 1))
            rows = np.hstack([b, -A])
            try:
                return _vertices_from(rows, "float")
            except RuntimeError:
                # cddlib's double-description method is run in floating point for speed, but on a degenerate
                # polytope -- near-parallel or redundant constraints, which arise when the coverage set holds
                # payoffs that are equal to within the rounding applied above -- it aborts with "Numerical
                # inconsistency is found. Use the GMP exact arithmetic." Retrying in exact rational arithmetic
                # is the remedy cddlib itself prescribes, and it returns identical vertices on inputs the
                # float path handles. It is only reached on the rare degenerate case, so the usual cost is one
                # failed attempt rather than exact arithmetic throughout.
                #
                # A is rounded to 4 decimals before this point, so limiting denominators to 1e6 is lossless
                # here while keeping cddlib away from the enormous denominators an exact float conversion
                # would produce.
                rational = [[Fraction(value).limit_denominator(10**6) for value in row] for row in rows]
                return _vertices_from(rational, "fraction")

        vertices = compute_poly_vertices(A, b)
        corners = []
        for v in vertices:
            corner_weight = v[:-1]
            # Make sure the corner weight is positive and sum to 1
            corner_weight = np.abs(corner_weight)
            corner_weight /= corner_weight.sum()
            corners.append(corner_weight)

        return corners

    def is_dominated(self, value: np.ndarray) -> bool:
        """Checks if the value is dominated by any of the values in the CCS.

        Args:
            value: Value vector

        Returns:
            True if the value is dominated by any of the values in the CCS, False otherwise.
        """
        if len(self.ccs) == 0:
            return False
        for w in self.visited_weights:
            if np.dot(value, w) >= self.max_scalarized_value(w):
                return False
        return True


if __name__ == "__main__":

    def _solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    num_objectives = 2
    ols = LinearSupport(num_objectives=num_objectives, epsilon=0.0001, verbose=True)
    w = ols.next_weight()
    while not ols.ended():
        print("w:", w)
        value = _solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(num_objectives), ols.ccs))
        w = ols.next_weight()

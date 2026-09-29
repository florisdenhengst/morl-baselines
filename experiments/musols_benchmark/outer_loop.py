"""Shared outer-loop harness for running an OLS/MUSOLS instance against a pluggable inner-loop solver."""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class OuterLoopResult:
    """Outcome of running an OLS/MUSOLS outer loop to convergence or until one of its budgets ran out."""

    ccs: List[np.ndarray]
    weight_support: List[np.ndarray]
    converged: bool
    elapsed_seconds: float
    num_evaluated: int
    # Per-iteration snapshots, present only when requested. Each entry records the coverage set as it stood
    # after that iteration, which is what turns these anytime algorithms into an anytime *curve*: quality
    # against evaluations spent, rather than a single end-of-run number.
    trajectory: Optional[List[dict]] = None


def run_outer_loop(
    algo,
    solver,
    max_seconds: float,
    max_evaluations: Optional[int] = None,
    record_trajectory: bool = False,
) -> OuterLoopResult:
    """Runs the OLS/MUSOLS outer loop against `solver`, bounded by wall-clock time and/or evaluation count.

    OLS and MUSOLS are anytime algorithms, so a budget is always needed. Wall-clock is the robust general
    bound: per-iteration cost is not uniform (corner-weight recomputation grows combinatorially with CCS size,
    while RL training cost is roughly fixed per policy), so an iteration count alone does not bound runtime.

    `max_evaluations` additionally caps the number of inner-loop solves. Use it when a run must be *exactly*
    reproducible: a wall-clock bound makes the number of completed iterations depend on machine load, whereas
    an evaluation cap does not. It is also the standard way to give two search strategies an equal budget of
    the genuinely expensive resource (policy evaluations) in a paper.

    Not converging within a budget is a valid, informative outcome (reported via `OuterLoopResult.converged`),
    not an error.

    Args:
        algo: A `LinearSupport` or `MUSOLS` instance.
        solver: An inner-loop solver with a `solve(w) -> np.ndarray` method.
        max_seconds: Wall-clock budget, in seconds.
        max_evaluations: Optional cap on inner-loop solves. None means only the wall-clock bound applies.
        record_trajectory: If True, snapshot the coverage set after every iteration, for anytime curves.

    Returns:
        OuterLoopResult
    """
    start = time.perf_counter()
    w = algo.next_weight()
    num_evaluated = 0
    trajectory: Optional[List[dict]] = [] if record_trajectory else None
    while not algo.ended() and time.perf_counter() - start < max_seconds:
        if max_evaluations is not None and num_evaluated >= max_evaluations:
            break
        value = solver.solve(w)
        algo.add_solution(value, w)
        num_evaluated += 1
        if trajectory is not None:
            trajectory.append(
                {
                    "num_evaluated": num_evaluated,
                    "elapsed_seconds": time.perf_counter() - start,
                    "ccs": [np.asarray(v).tolist() for v in algo.ccs],
                }
            )
        w = algo.next_weight()
    return OuterLoopResult(
        ccs=list(algo.ccs),
        weight_support=algo.get_weight_support(),
        converged=algo.ended(),
        elapsed_seconds=time.perf_counter() - start,
        num_evaluated=num_evaluated,
        trajectory=trajectory,
    )

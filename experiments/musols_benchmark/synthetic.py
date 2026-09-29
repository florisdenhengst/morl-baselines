"""A synthetic single-decision MOMDP for testing the scalability claims in isolation.

Why a synthetic task
---------------------
The theory predicts how the outer loop scales in the number of objectives d and the number of users m
(corner weights bounded via d' = min(d, m)). Measuring that on the real benchmarks is confounded twice over:
their d is fixed and small (2-6, so d cannot be swept at all), and on the RL-backed ones the inner-loop
training cost dwarfs the outer-loop cost, which is precisely the quantity under test. This task removes both
confounds -- d, m and |CCS| are all set directly, and the inner loop is an exact argmax with no learning -- so
what is measured is outer-loop behaviour and nothing else. It is a deliberate complement to, not a substitute
for, the accepted benchmarks: scalability claims are tested here, and the claims about real problems are
tested there.

The environment is a genuine (if degenerate) MOMDP: one state, N actions, one step, deterministic vector
reward. Every policy is exactly one of N fixed payoff vectors, which is the abstraction OLS-style theory
operates on anyway -- the CCS geometry, not the dynamics, is what drives the outer loop. Randomly generated
MOMDPs of this kind are the standard vehicle for scaling claims in the linear-support literature (Roijers,
2016, PhD thesis).

Candidate geometry
-------------------
  - "sphere" (default): payoffs are drawn on the positive orthant of the unit sphere, so no payoff is a convex
    combination of the others and *every* candidate lies on the convex hull. |CCS| is then exactly N by
    construction, which is what makes the corner-weight bound directly testable: (|CCS|, d, m) are all known
    rather than measured. It is also the worst case for the outer loop, which is the honest setting for a
    scalability stress test.
  - "gaussian": i.i.d. standard normal payoffs, giving the mix of dominated and non-dominated points typical
    of a real problem. |CCS| is then an outcome rather than a control, so this mode is for checking that
    conclusions are not an artifact of the worst-case geometry.
"""

from typing import List, Optional

import gymnasium as gym
import numpy as np
from environments import ENVIRONMENTS, EnvironmentConfig
from gymnasium.spaces import Box, Discrete


ENV_ID = "musols-synthetic-v0"
GEOMETRIES = ("sphere", "gaussian")


def generate_candidates(num_objectives: int, num_candidates: int, geometry: str, seed: int) -> np.ndarray:
    """Generates the fixed set of attainable payoff vectors.

    Args:
        num_objectives: Dimensionality d of each payoff vector.
        num_candidates: Number of attainable payoffs N (one per action).
        geometry: "sphere" (all N on the convex hull, so |CCS| = N) or "gaussian" (realistic mix).
        seed: Seed for the draw, making the task itself reproducible.

    Returns:
        np.ndarray of shape (num_candidates, num_objectives).
    """
    assert geometry in GEOMETRIES, f"geometry must be one of {GEOMETRIES}, got {geometry!r}."
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((num_candidates, num_objectives))
    if geometry == "sphere":
        points = np.abs(points)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points.astype(np.float32)


class SyntheticMOMDP(gym.Env):
    """One state, N actions, one step: taking action i yields payoff vector i and terminates."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_objectives: int = 4,
        num_candidates: int = 30,
        geometry: str = "sphere",
        candidate_seed: int = 0,
        render_mode: Optional[str] = None,
    ):
        """Initialize the synthetic MOMDP.

        Args:
            num_objectives: Dimensionality d of the reward vector.
            num_candidates: Number of actions / attainable payoff vectors N.
            geometry: Candidate geometry, see `generate_candidates`.
            candidate_seed: Seed fixing which payoff vectors this instance offers.
            render_mode: Unused; accepted for Gymnasium API compatibility.
        """
        self.render_mode = render_mode
        self.candidates = generate_candidates(num_objectives, num_candidates, geometry, candidate_seed)
        self.reward_dim = num_objectives
        self.observation_space = Discrete(1)
        self.action_space = Discrete(num_candidates)
        self.reward_space = Box(
            low=self.candidates.min(axis=0),
            high=self.candidates.max(axis=0),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets to the single state."""
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        """Returns the chosen candidate's payoff and terminates."""
        return 0, self.candidates[int(action)], True, False, {}

    def pareto_front(self, gamma: float) -> List[np.ndarray]:
        """Returns every attainable payoff vector (the exact solver and ground-truth metrics consume this).

        Args:
            gamma: Ignored; episodes are a single step, so no discounting applies.
        """
        return list(self.candidates)


gym.register(id=ENV_ID, entry_point=f"{__name__}:SyntheticMOMDP", disable_env_checker=True)


def synthetic_key(num_objectives: int) -> str:
    """Returns the `ENVIRONMENTS` key used for the synthetic task at a given number of objectives."""
    return f"synthetic-d{num_objectives}"


def register_synthetic_configs(
    num_objectives_list: List[int],
    num_candidates: int = 30,
    geometry: str = "sphere",
    candidate_seed: int = 0,
    epsilon: float = 0.001,
    timeout_seconds: float = 300.0,
) -> List[str]:
    """Registers one synthetic `EnvironmentConfig` per requested number of objectives.

    The registered `stakeholder_weights` are only a placeholder that satisfies `EnvironmentConfig`'s
    validation: every study cell overrides them with a freshly sampled panel (see `preferences.py`).

    Args:
        num_objectives_list: The values of d to make available.
        num_candidates: Number of attainable payoffs N per task.
        geometry: Candidate geometry, see `generate_candidates`.
        candidate_seed: Seed fixing the payoff vectors.
        epsilon: OLS/MUSOLS epsilon for these tasks.
        timeout_seconds: Per-algorithm wall-clock budget for these tasks.

    Returns:
        The registered environment keys, in the order requested.
    """
    keys = []
    for num_objectives in num_objectives_list:
        key = synthetic_key(num_objectives)
        ENVIRONMENTS[key] = EnvironmentConfig(
            key=key,
            env_id=ENV_ID,
            env_kwargs={
                "num_objectives": num_objectives,
                "num_candidates": num_candidates,
                "geometry": geometry,
                "candidate_seed": candidate_seed,
            },
            num_objectives=num_objectives,
            gamma=1.0,
            objective_names=[f"obj{i}" for i in range(num_objectives)],
            stakeholder_names=["placeholder-a", "placeholder-b"],
            stakeholder_weights=np.eye(num_objectives, dtype=np.float32)[:, :2],
            solver="exact",
            epsilon=epsilon,
            timeout_seconds=timeout_seconds,
            description=(
                f"Synthetic single-decision MOMDP with d={num_objectives}, N={num_candidates} attainable "
                f"payoffs ({geometry} geometry). Isolates outer-loop scaling from inner-loop learning cost."
            ),
        )
        keys.append(key)
    return keys

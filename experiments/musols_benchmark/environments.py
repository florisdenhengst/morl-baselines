"""Registry of benchmark environments for the MUSOLS-vs-OLS experiments.

Each environment models a single execution of actions (one episode) whose outcome must balance the
preferences of m=2 stakeholders with overlapping but distinct priorities -- see paper.latex for the general
motivation (healthcare, robotics, smart grid, ...: single-execution, multi-stakeholder decision making).

Each `EnvironmentConfig` fully specifies an experiment: the mo-gymnasium environment, its objectives, the
stakeholder preference matrix W (shape (num_objectives, 2)), which inner-loop solver to pair OLS/MUSOLS with,
and reasonable default hyperparameters. `run_experiment.py` reads from this registry and allows overriding any
of these defaults from the command line.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from gymnasium.wrappers import FlattenObservation
from reward_normalization import (
    lunar_lander_reward_scaler,
    water_reservoir_reward_scaler,
)


@dataclass
class EnvironmentConfig:
    """Full specification of one MUSOLS-vs-OLS benchmark environment."""

    key: str
    env_id: str
    env_kwargs: dict
    num_objectives: int
    gamma: float
    objective_names: List[str]
    stakeholder_names: List[str]
    stakeholder_weights: np.ndarray  # shape (num_objectives, 2): columns are the two stakeholders' weights.
    solver: str  # "exact" | "tabular_q" | "sac" | "discrete_sac"
    epsilon: float
    timeout_seconds: float
    description: str
    solver_kwargs: dict = field(default_factory=dict)
    # Optional env -> env callable applied before the reward wrapper, for environments whose raw observation
    # needs reshaping before any solver can consume it (e.g. highway's 2-D kinematics matrix).
    observation_wrapper: Optional[Callable[[object], object]] = None
    # Optional env -> env callable applying static, per-objective reward normalization (see
    # reward_normalization.py) for environments whose raw objectives have mismatched scales. None means the
    # raw environment rewards are already comparably scaled and need no adjustment.
    reward_wrapper: Optional[Callable[[object], object]] = None

    def __post_init__(self):
        assert self.stakeholder_weights.shape == (self.num_objectives, 2), (
            f"{self.key}: stakeholder_weights must have shape (num_objectives, 2), " f"got {self.stakeholder_weights.shape}."
        )
        assert np.allclose(
            self.stakeholder_weights.sum(axis=0), 1.0, atol=1e-4
        ), f"{self.key}: each stakeholder's weights must sum to 1."


ENVIRONMENTS: Dict[str, EnvironmentConfig] = {}


def _register(config: EnvironmentConfig) -> None:
    ENVIRONMENTS[config.key] = config


_register(
    EnvironmentConfig(
        key="fruit-tree",
        env_id="fruit-tree-v0",
        env_kwargs={"depth": 6},
        num_objectives=6,
        gamma=1.0,
        objective_names=["Protein", "Carbs", "Fats", "Vitamins", "Minerals", "Water"],
        stakeholder_names=["athlete", "dietitian"],
        # An athlete prioritizes protein and carbs (muscle recovery and energy); a dietitian prioritizes
        # vitamins and minerals (general nutritional balance). Both still care about every nutrient somewhat.
        stakeholder_weights=np.array(
            [
                [0.35, 0.10],  # Protein
                [0.30, 0.10],  # Carbs
                [0.05, 0.05],  # Fats
                [0.10, 0.30],  # Vitamins
                [0.10, 0.30],  # Minerals
                [0.10, 0.15],  # Water
            ],
            dtype=np.float32,
        ),
        solver="exact",
        epsilon=0.1,
        timeout_seconds=20.0,
        description=(
            "Healthcare/diet planning (Yang et al., 2019, https://arxiv.org/abs/1908.08342). A single meal "
            "plan (one root-to-leaf walk of a binary decision tree) balances the differing nutrient "
            "priorities of an athlete and a dietitian."
        ),
    )
)

_register(
    EnvironmentConfig(
        key="deep-sea-treasure",
        env_id="deep-sea-treasure-v0",
        env_kwargs={},
        num_objectives=2,
        gamma=1.0,
        objective_names=["treasure", "time_penalty"],
        stakeholder_names=["treasure hunter", "operations manager"],
        # A treasure hunter mostly cares about the treasure collected; an operations manager mostly cares
        # about time/fuel efficiency, but still wants some treasure recovered.
        stakeholder_weights=np.array(
            [
                [0.90, 0.40],  # treasure
                [0.10, 0.60],  # time_penalty
            ],
            dtype=np.float32,
        ),
        solver="exact",
        epsilon=0.01,
        timeout_seconds=10.0,
        description=(
            "The classic Deep Sea Treasure benchmark (Vamplew et al., 2011, via Yang et al., 2019). A single "
            "submarine dive (one episode) balances a treasure hunter's and an operations manager's differing "
            "priorities between treasure value and time efficiency."
        ),
    )
)

_register(
    EnvironmentConfig(
        key="resource-gathering",
        env_id="resource-gathering-v0",
        env_kwargs={},
        num_objectives=3,
        gamma=0.95,
        objective_names=["enemy_penalty", "gold", "gem"],
        stakeholder_names=["safety officer", "logistics manager"],
        # A safety officer weighs the risk of encountering enemies far more heavily; a logistics manager
        # weighs resource collection more heavily. Both care about all three objectives to some degree.
        stakeholder_weights=np.array(
            [
                [0.70, 0.15],  # enemy_penalty
                [0.20, 0.55],  # gold
                [0.10, 0.30],  # gem
            ],
            dtype=np.float32,
        ),
        solver="tabular_q",
        solver_kwargs={
            "total_timesteps": 30_000,
            "learning_rate": 0.3,
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_decay_frac": 0.5,
        },
        epsilon=0.05,
        timeout_seconds=120.0,
        description=(
            "Robotics: a single robot run (Barrett & Narayanan, 2008), gathering resources and returning "
            "home, balancing a safety officer's and a logistics manager's differing risk/reward priorities. "
            "Trains a real tabular Q-learning policy per candidate weight."
        ),
    )
)

_register(
    EnvironmentConfig(
        key="minecart",
        env_id="minecart-v0",
        env_kwargs={},
        num_objectives=3,
        gamma=0.98,  # the discount this benchmark is conventionally reported at (cf. examples/gpi_pd_minecart.py)
        objective_names=["ore_1", "ore_2", "fuel_cost"],
        stakeholder_names=["ore-A buyer", "ore-B buyer"],
        # Two downstream buyers share one cart and one fuel budget: each wants their own mineral, and both
        # carry the fuel cost equally. A genuine two-party resource conflict rather than a contrived split.
        stakeholder_weights=np.array(
            [
                [0.55, 0.20],  # ore_1
                [0.20, 0.55],  # ore_2
                [0.25, 0.25],  # fuel_cost
            ],
            dtype=np.float32,
        ),
        solver="discrete_sac",
        solver_kwargs={
            "total_timesteps": 100_000,
            "net_arch": [64, 64],
            "learning_starts": 1_000,
            "batch_size": 128,
        },
        epsilon=0.05,
        timeout_seconds=3600.0,
        description=(
            "Mining logistics (Abels et al., 2019). A single cart trip balances two ore buyers' competing "
            "demands against a shared fuel budget. Widely used MORL benchmark with a known Pareto front. "
            "Trains a real discrete-action SAC policy per candidate weight."
        ),
        # No reward normalization: on the environment's own known Pareto front the three objectives span
        # comparable ranges ([0.92, 0.92, 0.87], ratio 1.1), so the raw scales are already commensurate.
        # (Random-policy returns look wildly imbalanced only because a random agent never completes the
        # sparse mine-and-return task; achievable ranges, not random rollouts, are the right reference.)
    )
)

_register(
    EnvironmentConfig(
        key="highway",
        # The "fast" variant is highway-env's training-optimized configuration (fewer vehicles, lower
        # simulation frequency) and is what the RL literature trains on. It is not a cosmetic choice here:
        # measured on this machine, mo-highway-v0 costs 90.6 ms per environment step against 8.3 ms for this
        # one -- 11x, or 75 minutes of pure simulation per policy at 50k steps before a single network update,
        # which makes the full-fidelity variant impractical across four algorithms and many seeds.
        env_id="mo-highway-fast-v0",
        env_kwargs={},
        num_objectives=3,
        gamma=0.99,
        objective_names=["speed", "right_lane", "collision"],
        stakeholder_names=["passenger", "fleet safety officer"],
        # The paper's motivating self-driving scenario: the rider wants to arrive quickly, the operator wants
        # lane discipline and above all no collisions. Both care about all three, with opposed emphasis.
        stakeholder_weights=np.array(
            [
                [0.55, 0.15],  # speed
                [0.10, 0.35],  # right_lane
                [0.35, 0.50],  # collision
            ],
            dtype=np.float32,
        ),
        solver="discrete_sac",
        solver_kwargs={
            "total_timesteps": 50_000,
            "net_arch": [64, 64],
            "learning_starts": 1_000,
            "batch_size": 128,
        },
        epsilon=0.05,
        timeout_seconds=1800.0,
        description=(
            "Autonomous driving (Leurent, 2018, highway-env). A single highway run balances a passenger's "
            "preference for speed against a fleet operator's preference for lane discipline and collision "
            "avoidance. Trains a real discrete-action SAC policy per candidate weight."
        ),
        # highway-env emits a (5, 5) kinematics matrix; flatten it so the solver sees a plain feature vector.
        observation_wrapper=FlattenObservation,
    )
)

_register(
    EnvironmentConfig(
        key="reacher",
        env_id="mo-reacher-v5",
        env_kwargs={},
        num_objectives=4,
        gamma=0.99,
        objective_names=["target_1", "target_2", "target_3", "target_4"],
        stakeholder_names=["operator A", "operator B"],
        # Four competing targets for one arm: two operators each own a different pair of targets, so neither
        # can be served without giving something up. All four objectives share the form r_i = 1 - 4*d_i^2, so
        # they are on the same scale by construction and need no normalization.
        stakeholder_weights=np.array(
            [
                [0.35, 0.15],  # target_1
                [0.35, 0.15],  # target_2
                [0.15, 0.35],  # target_3
                [0.15, 0.35],  # target_4
            ],
            dtype=np.float32,
        ),
        solver="discrete_sac",
        solver_kwargs={
            "total_timesteps": 50_000,
            "net_arch": [64, 64],
            "learning_starts": 1_000,
            "batch_size": 128,
        },
        epsilon=0.05,
        timeout_seconds=1800.0,
        description=(
            "Robotic arm control (MuJoCo Reacher, as multi-objectivized in mo-gymnasium). A single reach "
            "balances four competing target locations owned by two operators. Trains a real discrete-action "
            "SAC policy per candidate weight."
        ),
    )
)

_register(
    EnvironmentConfig(
        key="water-reservoir",
        env_id="water-reservoir-v0",
        env_kwargs={"nO": 4, "normalized_action": True},
        num_objectives=4,
        gamma=0.99,
        objective_names=["upstream_flood_cost", "water_supply_deficit", "hydro_deficit", "downstream_flood_cost"],
        stakeholder_names=["flood control engineer", "utility operator"],
        # A flood control engineer's mandate is almost exclusively flood prevention; a utility operator's is
        # almost exclusively meeting supply/hydro demand. Each still assigns a small (0.05) residual weight to
        # the other's concerns rather than zero, keeping this non-one-hot, but more sharply differentiated than
        # a 0.40/0.10 split -- needed here because the downstream-flood objective's magnitude, even after
        # normalization, is large enough that a 0.10 residual weight on it alone can dominate the utility
        # operator's entire scalarized value (see reward_normalization.py's water-reservoir discussion).
        stakeholder_weights=np.array(
            [
                [0.45, 0.05],  # upstream_flood_cost
                [0.05, 0.45],  # water_supply_deficit
                [0.05, 0.45],  # hydro_deficit
                [0.45, 0.05],  # downstream_flood_cost
            ],
            dtype=np.float32,
        ),
        solver="sac",
        solver_kwargs={
            # Water-reservoir's continuous release-control problem (stochastic inflow, sparse-ish trade-off
            # signal) needs a substantially larger SAC budget than lunar-lander before its policy starts
            # responding to the objective weighting at all: at 500-8,000 steps, policies trained on
            # near-opposite weight vectors converged to nearly identical behavior (see
            # reward_normalization.py). 75,000 steps is a middle estimate of the typical budget for SAC to
            # converge on an environment of this kind.
            "total_timesteps": 75_000,
            "net_arch": [64, 64],
            "learning_starts": 256,
            "batch_size": 128,
        },
        epsilon=0.01,
        timeout_seconds=1800.0,
        description=(
            "Smart grid / water management (Castelletti et al., 2012, IJCNN). A single reservoir-operation "
            "episode balances a flood control engineer's and a utility operator's differing priorities "
            "between flood risk and supply reliability. Trains a real continuous-control SAC policy per "
            "candidate weight."
        ),
        # The four objectives have very different physical units (a reservoir-level excess vs. a
        # demand-shortfall deficit) and, left raw, wildly different magnitudes; without normalization one or
        # two objectives dominate every weight vector regardless of the assigned stakeholder preferences.
        # Normalized here via the environment's own documented ideal/nadir reference points (see
        # reward_normalization.py).
        reward_wrapper=water_reservoir_reward_scaler,
    )
)

_register(
    EnvironmentConfig(
        key="lunar-lander",
        env_id="mo-lunar-lander-continuous-v3",
        env_kwargs={},
        num_objectives=4,
        gamma=0.99,
        objective_names=["landing_success", "shaping", "main_fuel_cost", "side_fuel_cost"],
        stakeholder_names=["safety officer", "fuel/operations manager"],
        # A safety officer prioritizes landing successfully with a smooth, shaped descent; a fuel/operations
        # manager prioritizes minimizing fuel cost, while both still care about a safe landing.
        stakeholder_weights=np.array(
            [
                [0.55, 0.30],  # landing_success
                [0.30, 0.10],  # shaping
                [0.10, 0.35],  # main_fuel_cost
                [0.05, 0.25],  # side_fuel_cost
            ],
            dtype=np.float32,
        ),
        solver="sac",
        solver_kwargs={
            "total_timesteps": 8_000,
            "net_arch": [64, 64],
            "learning_starts": 256,
            "batch_size": 128,
        },
        epsilon=0.05,
        timeout_seconds=900.0,
        description=(
            "Continuous control (self-driving/vehicle-routing analog): a single lunar landing (one episode) "
            "balances a safety officer's and a fuel/operations manager's differing priorities between landing "
            "safely and conserving fuel. Trains a real continuous-control SAC policy per candidate weight."
        ),
        # Terminal landing/crash reward and cumulative shaping reward are both on a ~100 scale by the
        # environment's design; fuel costs are raw per-step engine draws accumulating to a much smaller
        # magnitude. Static division brings all four objectives to a comparable scale (see
        # reward_normalization.py).
        reward_wrapper=lunar_lander_reward_scaler,
    )
)

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
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from reward_normalization import (
    highway_reward_scaler,
    hopper_reward_scaler,
    lunar_lander_reward_scaler,
    minecart_reward_scaler,
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
            "total_timesteps": 500_000,
            "net_arch": [64, 64],
            # Minecart charges fuel densely every step but pays ore only on returning to base. Without a
            # substantial pure-random warmup, SAC sees almost nothing but early negative fuel reward and
            # collapses onto "do as little as possible" before it has ever observed a completed mine-and-return
            # episode. 10k steps of uniform action sampling first makes such an episode likely to be in the
            # buffer at all by the time learning starts.
            "learning_starts": 10_000,
            "batch_size": 128,
            # Large enough that those rare early successful mining episodes are still sampleable late in the
            # run rather than having been evicted by a long tail of unsuccessful ones.
            "buffer_size": 500_000,
        },
        epsilon=0.05,
        timeout_seconds=3600.0,
        description=(
            "Mining logistics (Abels et al., 2019). A single cart trip balances two ore buyers' competing "
            "demands against a shared fuel budget. Widely used MORL benchmark with a known Pareto front. "
            "Trains a real discrete-action SAC policy per candidate weight."
        ),
        # Normalized against the environment's own known Pareto front. The achievable spans are already within
        # 6% of each other, so this is near-identity in magnitude -- it is applied so the common scale is
        # explicit and checked rather than incidental. See reward_normalization.py, which also explains why
        # this does not by itself address minecart's real difficulty (reward *density*, not magnitude).
        reward_wrapper=minecart_reward_scaler,
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
            "total_timesteps": 500_000,
            "net_arch": [64, 64],
            "learning_starts": 5_000,
            "batch_size": 128,
            "buffer_size": 500_000,
            # Collisions end episodes immediately, so aggressive early exploration fills the buffer with very
            # short crash trajectories and little else. MOSACDiscrete's autotuned entropy coefficient starts at
            # exp(0) = 1.0 -- near-uniform lane swapping -- and the `alpha` argument is ignored while autotune
            # is on (see solvers.py). Disabling autotune is therefore the only way to actually start
            # conservative, at an entropy weight low enough that the policy follows its Q-values early on.
            "autotune": False,
            "alpha": 0.05,
        },
        epsilon=0.05,
        timeout_seconds=1800.0,
        description=(
            "Autonomous driving (Leurent, 2018, highway-env). A single highway run balances a passenger's "
            "preference for speed against a fleet operator's preference for lane discipline and collision "
            "avoidance. Trains a real discrete-action SAC policy per candidate weight."
        ),
        # Stack 4 consecutive kinematics frames before flattening: a single (5, 5) frame gives absolute vehicle
        # positions with no history, so relative velocities -- the quantity that actually decides whether a gap
        # is closing -- have to be inferred from one snapshot. Stacking makes them directly observable and is
        # standard practice for this benchmark. The result is a (4, 5, 5) tensor, flattened to 100 features.
        observation_wrapper=lambda env: FlattenObservation(FrameStackObservation(env, stack_size=4)),
        # Speed and right-lane are dense per-step rewards; collision is a one-time -1. Over the environment's
        # 30-step episode the dense pair outruns the collision penalty by an order of magnitude, so without
        # rescaling SAC learns to drive flat out and absorb crashes whatever weight safety is given. See
        # reward_normalization.py for the measured per-step rates behind the divisors.
        reward_wrapper=highway_reward_scaler,
    )
)

_register(
    EnvironmentConfig(
        key="reacher",
        env_id="mo-reacher-v5",
        env_kwargs={},
        num_objectives=4,
        # Reacher episodes reset after ~50 steps, so there is no long horizon to discount for; 0.98 gives an
        # effective horizon that matches the episode rather than reaching well past its end as 0.99 does.
        gamma=0.98,
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
            "total_timesteps": 500_000,
            "net_arch": [64, 64],
            "learning_starts": 1_000,
            "batch_size": 128,
            # The state-action space here is small and episodes are short, so 100k transitions already covers
            # a wide range of arm configurations; a larger buffer would only add memory, not diversity.
            "buffer_size": 100_000,
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
        key="hopper",
        env_id="mo-hopper-v5",
        env_kwargs={},
        num_objectives=3,
        # Cyclic locomotion needs a long horizon: the trade-off between an energetic push-off now and staying
        # balanced (and therefore alive, still collecting reward) many steps later only appears at a high
        # discount. 0.99 over the 1000-step episode limit.
        gamma=0.99,
        objective_names=["forward_velocity", "hop_height", "energy_saving"],
        stakeholder_names=["logistics operator", "maintenance engineer"],
        # A delivery operator wants throughput (distance covered per episode); a maintenance engineer wants
        # low actuator wear. Both care about hop height, which stands in for gait stability -- a hopper that
        # stops clearing the ground is about to fall over, which serves neither party.
        #
        # The maintenance stakeholder's energy weight is deliberately capped at 0.50 rather than pushed
        # higher: because `healthy_reward` is added to every objective, an agent that does nothing scores the
        # *maximum* +1/step on the energy objective, so an energy-dominant weight vector makes standing still
        # optimal and the episode degenerates. See reward_normalization.py.
        stakeholder_weights=np.array(
            [
                [0.60, 0.25],  # forward_velocity
                [0.25, 0.25],  # hop_height
                [0.15, 0.50],  # energy_saving
            ],
            dtype=np.float32,
        ),
        solver="sac",
        solver_kwargs={
            "total_timesteps": 1_000_000,
            "net_arch": [256, 256],
            # The 3-link hopper is inherently unstable and terminates the moment the torso tilts too far or
            # the height drops below threshold, so early episodes are extremely short. 10k steps of uniform
            # action sampling gives the Q-function a broad spread of joint configurations to fit before any
            # gradient update, rather than a buffer of near-identical immediate-failure trajectories.
            "learning_starts": 10_000,
            "batch_size": 256,
            # Full 1M transitions: a hopping gait is only learnable by contrasting the early unstable steps
            # against full steady-state strides acquired much later, so early data must stay sampleable.
            "buffer_size": 1_000_000,
            # Automatic entropy tuning works well on this task, and MOSAC's autotune happens to start exactly
            # where we want it: log_alpha is initialized to zeros, so alpha begins at exp(0) = 1.0 -- high
            # enough that the policy keeps exploring rather than committing early to a lopsided, asymmetric
            # leg-extension pattern, then anneals itself down. (Contrast highway, where that same high start
            # is harmful and autotune is therefore disabled.) `alpha` is ignored while autotune is on.
            "autotune": True,
        },
        epsilon=0.05,
        timeout_seconds=1800.0,
        description=(
            "Continuous-control locomotion (MuJoCo Hopper, as multi-objectivized in mo-gymnasium). A single "
            "hopping run balances a logistics operator's demand for throughput against a maintenance "
            "engineer's demand for low actuator wear, with gait stability shared between them. Trains a real "
            "continuous-control SAC policy per candidate weight."
        ),
        # All three objectives carry the same +1/step healthy_reward, and the raw energy objective swings by 3
        # per step -- as much as forward velocity -- so without rescaling an energy-weighted stakeholder
        # prefers standing still to hopping. See reward_normalization.py for the structural bounds used.
        reward_wrapper=hopper_reward_scaler,
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
            "total_timesteps": 500_000,
            "net_arch": [64, 64],
            # Early on, fuel penalties and crash rewards are all the agent sees, and SAC will happily collapse
            # onto cutting the engines and falling passively -- locally the cheapest option -- before it has
            # ever seen a successful landing. 10k steps of pure random action sampling first makes soft
            # landings present in the buffer when learning begins.
            "learning_starts": 10_000,
            "batch_size": 128,
            "buffer_size": 500_000,
        },
        epsilon=0.05,
        timeout_seconds=900.0,
        description=(
            "Continuous control (self-driving/vehicle-routing analog): a single lunar landing (one episode) "
            "balances a safety officer's and a fuel/operations manager's differing priorities between landing "
            "safely and conserving fuel. Trains a real continuous-control SAC policy per candidate weight."
        ),
        # All four objectives are divided by 100, bringing episode returns to an O(1) range while keeping the
        # relative proportions the environment itself assigns them. The fuel pair previously used a divisor of
        # 25, which inflated fuel to ~1.9x the magnitude of landing success and so made crashing promptly look
        # better than firing the engines -- see reward_normalization.py for the measured returns behind this.
        reward_wrapper=lunar_lander_reward_scaler,
    )
)

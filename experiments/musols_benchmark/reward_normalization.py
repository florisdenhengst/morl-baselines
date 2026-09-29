"""Static, per-objective reward normalization for environments with mismatched objective scales.

Why this exists
----------------
Linear scalarization (u(v) = w^T v, used throughout OLS/MUSOLS) implicitly assumes the objectives in v are on
comparable scales: a weight of 0.4 is only meaningful as "40% priority" if a one-unit change in each objective
represents a roughly comparable amount of outcome. When objectives differ by orders of magnitude (as with
lunar-lander's terminal reward vs. its per-step fuel costs, or water-reservoir's very differently-united flood
and deficit costs), the weight vector is dominated by whichever objective happens to have the largest raw
magnitude, regardless of the assigned weight -- silently defeating the whole point of specifying stakeholder
preferences. This is a well-documented pitfall in the MORL literature (see e.g. Hayes et al., 2022, "A
practical guide to multi-objective reinforcement learning and planning", Section on scalarization functions
and objective normalization) and the standard fix is the same one used broadly in RL for reward scaling more
generally (e.g. reward clipping/normalization in DQN, PPO, and `VecNormalize`-style wrappers): rescale each
reward component to a comparable range *before* it reaches the learning algorithm.

Design principle: static, environment-intrinsic, policy-independent
---------------------------------------------------------------------
Every scaling factor here is a fixed constant derived from an environment's own documented/physical reward
ranges -- never fitted from rollout statistics of any particular policy. This avoids two unreasonable
assumptions: (1) that we already know the true Pareto front (we don't -- that's what OLS/MUSOLS is for), and
(2) that normalization should adapt per-policy or per-weight, which would distort the trade-off geometry
differently for different stakeholders and make the resulting CCS incomparable across weights. A fixed,
policy-independent scale preserves consistent relative trade-off ratios across the whole weight simplex.

Two normalization techniques are provided, matched to what reference information is available:
  - `PerObjectiveRewardScaler` (static linear division): appropriate when each objective's typical episode
    magnitude is known or can be reasonably estimated (e.g. lunar-lander's terminal/shaping reward is
    calibrated by the environment's own design to a ~100 scale). Dividing by that magnitude brings the
    objective to an O(1) scale while preserving a fixed, interpretable trade-off ratio.
  - `IdealNadirRewardScaler` (ideal-point/nadir-point normalization): the standard technique from
    multi-objective optimization (e.g. as used in NSGA-II-style algorithms) for objectives with very different
    physical units, where the environment itself documents a best-case ("ideal"/"utopia") and worst-case
    ("nadir"/"antiutopia") per-step reference value. Rescaling via r' = (r - ideal) / (ideal - nadir) maps
    every objective onto the same [-1, 0] per-step range using only these environment-intrinsic references --
    no per-objective "typical magnitude" needs to be estimated or guessed.

Both wrappers never modify the underlying environment or its raw rewards -- `env.step()` on the wrapped
environment returns the fully normalized reward, while `env.unwrapped` still exposes the original, untouched
environment (including its raw `pareto_front()`, if any).
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np


class PerObjectiveRewardScaler(gym.RewardWrapper):
    """Rescales each objective of a vector reward by a fixed, environment-intrinsic linear divisor.

    Objective indices not present in `linear_scale` are passed through unchanged. See the module docstring for
    the rationale and when to prefer this over `IdealNadirRewardScaler`.
    """

    def __init__(self, env: gym.Env, linear_scale: Optional[Dict[int, float]] = None):
        """Initialize the scaler.

        Args:
            env: The (mo-gymnasium) environment to wrap.
            linear_scale: Maps objective index -> fixed positive divisor.
        """
        super().__init__(env)
        self.linear_scale = dict(linear_scale or {})
        assert all(s > 0 for s in self.linear_scale.values()), "linear_scale divisors must be positive."

        base_space = env.unwrapped.reward_space
        low = np.array(base_space.low, dtype=np.float32)
        high = np.array(base_space.high, dtype=np.float32)
        for i, s in self.linear_scale.items():
            low[i] = low[i] / s
            high[i] = high[i] / s
        self.reward_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reward(self, reward: np.ndarray) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float32)
        out = reward.copy()
        for i, s in self.linear_scale.items():
            out[i] = reward[i] / s
        return out


class IdealNadirRewardScaler(gym.RewardWrapper):
    """Rescales each objective onto the standard [-1, 0] ideal-point/nadir-point range.

    r' = (r - ideal) / (ideal - nadir)

    `ideal` (best case) maps to 0 and `nadir` (worst case) maps to -1 for every objective, regardless of each
    objective's raw physical units or magnitude. See the module docstring for the rationale.
    """

    def __init__(self, env: gym.Env, ideal, nadir):
        """Initialize the scaler.

        Args:
            env: The (mo-gymnasium) environment to wrap.
            ideal: Per-objective best-case ("utopia") reference value.
            nadir: Per-objective worst-case ("antiutopia") reference value.
        """
        super().__init__(env)
        self.ideal = np.asarray(ideal, dtype=np.float32)
        self.nadir = np.asarray(nadir, dtype=np.float32)
        assert np.all(self.ideal >= self.nadir), "ideal must be at least as good (>=) as nadir for every objective."
        self._scale = self.ideal - self.nadir
        assert np.all(self._scale > 0), "ideal and nadir must differ for every objective."

        base_space = env.unwrapped.reward_space
        low = (np.array(base_space.low, dtype=np.float32) - self.ideal) / self._scale
        high = (np.array(base_space.high, dtype=np.float32) - self.ideal) / self._scale
        self.reward_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reward(self, reward: np.ndarray) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float32)
        return (reward - self.ideal) / self._scale


def lunar_lander_reward_scaler(env: gym.Env) -> PerObjectiveRewardScaler:
    """Static reward normalization for mo-lunar-lander-continuous-v3.

    Objectives (see mo_gymnasium's LunarLander docstring): [landing_success, shaping, main_fuel_cost,
    side_fuel_cost]. The terminal landing_success/crash reward and the cumulative shaping reward are both
    calibrated, by the original environment's design, to a +-100-ish scale (a "solved" episode totals ~200-300
    combining the two); the two fuel-cost objectives are raw, un-scaled per-step engine power draws.

    All four divisors are 100. The fuel pair previously used 25, which turned out to *invert* the intended
    priority rather than correct it: measured over random-policy episodes, the raw per-episode magnitudes are
    already comparable --

        landing_success -100      shaping -392..+126      main_fuel -85..-25      side_fuel -88..-25

    -- so dividing fuel by 25 while dividing landing by 100 inflated each fuel objective to roughly 1.9x the
    magnitude of landing success (mean |return| 1.88 and 1.91 against 1.00). That is precisely the failure mode
    where over-weighted fuel penalties make crashing promptly look preferable to firing the engines to save the
    lander. A common divisor keeps the four objectives in the relative proportions the environment itself
    assigns them, while still bringing episode returns to an O(1) range -- which also keeps SAC's critic targets
    near unit scale instead of +-100, where value regression is markedly less stable. This remains a fixed
    linear rescaling: relative trade-off ratios and Pareto/CCS geometry are constant across the whole simplex.
    """
    return PerObjectiveRewardScaler(env, linear_scale={0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0})


def minecart_reward_scaler(env: gym.Env) -> PerObjectiveRewardScaler:
    """Static reward normalization for minecart-v0, from its own known Pareto front.

    Objectives: [ore_1, ore_2, fuel_cost]. Measured over the environment's analytically known Pareto front at
    gamma=0.98, the achievable per-objective ranges are

        ore_1 [0, 0.9236]      ore_2 [0, 0.9236]      fuel_cost [-1.1184, -0.2492]

    i.e. spans of 0.9236 / 0.9236 / 0.8692 -- already within 6% of each other. Dividing by those spans is
    therefore close to an identity, and is applied for explicitness: the objectives sit on a common scale by
    construction rather than by an undocumented coincidence a future environment change could quietly break.

    Deliberately linear division rather than `IdealNadirRewardScaler`, even though a known front makes ideal
    and nadir available. Ideal/nadir normalization *shifts* as well as scales (r' = (r - ideal)/(ideal - nadir)),
    and these references are episode-return quantities while a RewardWrapper applies per step. Every ordinary
    step delivers no ore, so shifting would map that 0 to -1 on both ore objectives and manufacture a large
    dense penalty out of what is genuinely a sparse reward. (Water-reservoir can use ideal/nadir safely because
    DamEnv documents true *per-step* utopia/antiutopia references.) Pure division leaves sparse zeros at zero.

    Note what this does *not* fix. Minecart's training difficulty is a reward *density* mismatch, not a
    magnitude one: fuel is charged every step while ore is paid only on returning to base, so early training is
    dominated by negative fuel signal however the objectives are scaled. That is what the solver's
    random-exploration warmup (`learning_starts`) and a buffer large enough to retain early successful mining
    episodes are for -- see environments.py.
    """
    return PerObjectiveRewardScaler(env, linear_scale={0: 0.9236, 1: 0.9236, 2: 0.8692})


def highway_reward_scaler(env: gym.Env) -> PerObjectiveRewardScaler:
    """Static reward normalization for mo-highway-fast-v0.

    Objectives: [speed, right_lane, collision]. The first two are dense per-step rewards; collision is a
    one-time -1 that also ends the episode. With `duration = 30` (the environment's own config), a
    collision-free episode accumulates far more speed/lane reward than a crash ever costs -- measured per-step
    rates over random rollouts are ~0.38/step speed and ~0.54/step right_lane, so a full 30-step episode earns
    roughly +11.5 and +16.2 against a single -1.0 for crashing. Left raw, that is an order-of-magnitude
    incentive to drive flat out and absorb frequent high-speed collisions, whatever weight a stakeholder
    nominally places on safety.

    Dividing by those full-episode accumulations puts a complete, collision-free episode at about +1.0 on each
    dense objective, directly comparable in scale to the -1.0 collision penalty, so the stakeholder weights
    again decide the trade-off rather than the raw magnitudes.
    """
    return PerObjectiveRewardScaler(env, linear_scale={0: 12.0, 1: 16.0, 2: 1.0})


def water_reservoir_reward_scaler(env: gym.Env) -> IdealNadirRewardScaler:
    """Static reward normalization for water-reservoir-v0 (nO=4), using the environment's own documented
    per-step ideal/nadir reference points rather than an ad hoc per-objective scale factor.

    Objectives: [upstream_flood_cost, water_supply_deficit, hydro_deficit, downstream_flood_cost]. These have
    very different physical units (a reservoir-level excess vs. a demand-shortfall deficit), so no single
    "typical magnitude" divisor generalizes across all four the way it does for lunar-lander. mo_gymnasium's
    `DamEnv` (Castelletti et al., 2012), the source of this environment, already documents per-step best-case
    ("utopia") and worst-case ("antiutopia") reference values for this exact nO=4 configuration:
        ideal (utopia[4])     = [-0.5, -9.0, -0.001, -9.0]
        nadir (antiutopia[4]) = [-65.0, -12.0, -0.7, -12.0]
    Rescaling each objective via the standard ideal-point/nadir-point normalization (as in NSGA-II-style
    multi-objective optimization) maps every objective onto the same [-1, 0] per-step range using only these
    environment-intrinsic references -- no per-objective typical-magnitude estimate needs to be guessed.
    """
    ideal = [-0.5, -9.0, -0.001, -9.0]
    nadir = [-65.0, -12.0, -0.7, -12.0]
    return IdealNadirRewardScaler(env, ideal=ideal, nadir=nadir)

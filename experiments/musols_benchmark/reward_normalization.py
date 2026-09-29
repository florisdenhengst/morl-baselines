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
    combining the two); the two fuel-cost objectives are raw, un-scaled per-step engine power draws that
    accumulate to a much smaller magnitude over an episode. Dividing the terminal/shaping pair by 100 and the
    fuel-cost pair by 25 (their typical empirical episode totals) brings all four objectives to a comparable
    O(1) scale. This is a fixed linear rescaling: it retains constant relative trade-off ratios and consistent
    Pareto/CCS geometry across every weight in the simplex.
    """
    return PerObjectiveRewardScaler(env, linear_scale={0: 100.0, 1: 100.0, 2: 25.0, 3: 25.0})


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

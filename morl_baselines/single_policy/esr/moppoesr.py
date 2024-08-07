"""EUPG is an ESR algorithm based on Policy Gradient (REINFORCE like)."""
import time
from copy import deepcopy
from typing import Callable, List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from morl_baselines.common.value_prob_accrued_reward_buffer import ValueProbAccruedRewardReplayBuffer
# TODO FdH: remove unused import replay buffer
from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp


class PolicyNet(nn.Module):
    """Policy network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the policy network.

        Args:
            obs_shape: Observation shape
            action_dim: Action dimension
            rew_dim: Reward dimension
            net_arch: Number of units per layer
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim

        # Conditioned on accrued reward, so input takes reward
        input_dim = obs_shape[0] + rew_dim

        # |S|+|R| -> ... -> |A|
        self.actor = mlp(input_dim, action_dim, net_arch, activation_fn=nn.Tanh)
        self.critic = mlp(input_dim, 1, net_arch, activation_fn=nn.Tanh)
        self.apply(layer_init)
    
    def get_value(self, obs:th.Tensor, acc_reward: th.Tensor):
        input = th.cat((obs, acc_reward), dim=acc_reward.dim() - 1)
        return self.critic(input)

    def forward(self, obs: th.Tensor, acc_reward: th.Tensor):
        """Forward pass of actor.

        Args:
            obs: Observation
            acc_reward: accrued reward

        Returns: Probability of each action

        """
        input = th.cat((obs, acc_reward), dim=acc_reward.dim() - 1)
        pi = self.actor(input)
        # Normalized sigmoid
        x_exp = th.sigmoid(pi)
        probas = x_exp / th.sum(x_exp)
        return probas.view(-1, self.action_dim)  # Batch Size x |Actions|

    def distribution(self, obs: th.Tensor, acc_reward: th.Tensor):
        """Categorical distribution based on the action probabilities of actor.

        Args:
            obs: observation
            acc_reward: accrued reward

        Returns: action distribution.

        """
        probas = self.forward(obs, acc_reward)
        distribution = Categorical(probas)
        return distribution


class MOPPOESR(MOPolicy, MOAgent):
    """Expected Utility Policy Gradient Algorithm.

    The idea is to condition the network on the accrued reward and to scalarize the rewards based on the episodic return (accrued + future rewards)
    Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.
    """

    def __init__(
        self,
        env: gym.Env,
        scalarization: Callable[[np.ndarray, np.ndarray], float],
        weights: np.ndarray = np.ones(2),
        id: Optional[int] = None,
        buffer_size: int = int(1e5),
        net_arch: List = [50],
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "EUPG",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 1000,
        device: Union[th.device, str] = "auto",
        #TODO FdH: check whether clip_range should be of Union[np.float32, Schedule] type
        max_grad_norm: float = 0.5,
        clip_range: Callable[[int], float] = None, # callable that returns the clipping range, could be set in [0.1, 0.3]
        ent_coef: float = 0.0, # advice to set in [0, 0.01] to regularize the policy
        vf_coef: float = 0.5,
        ppo_clip: Optional[bool] = True,
        use_advantage: Optional[bool] = True,
        clip_range_vf: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the EUPG algorithm.

        Args:
            env: Environment
            scalarization: Scalarization function to use (can be non-linear)
            weights: Weights to use for the scalarization function
            id: Id of the agent (for logging)
            buffer_size: Size of the replay buffer
            net_arch: Number of units per layer
            gamma: Discount factor
            learning_rate: Learning rate (alpha)
            project_name: Name of the project (for logging)
            experiment_name: Name of the experiment (for logging)
            wandb_entity: Entity to use for wandb
            log: Whether to log or not
            log_every: Log every n episodes
            device: Device to use for NN. Can be "cpu", "cuda" or "auto".
            seed: Seed for the random number generator
            parent_rng: Parent random number generator (for reproducibility)
        """
        MOAgent.__init__(self, env, device, seed=seed)
        MOPolicy.__init__(self, None, device)

        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        self.env = env
        self.id = id
        # RL
        self.scalarization = scalarization
        self.weights = weights
        self.gamma = gamma

        # Learning
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.buffer = ValueProbAccruedRewardReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
            obs_dtype=np.int32,
            action_dtype=np.int32,
        )
        self.net = PolicyNet(
            obs_shape=self.observation_shape,
            rew_dim=self.reward_dim,
            action_dim=self.action_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.ppo_clip = ppo_clip
        self.use_advantage = use_advantage
        self.clip_range_vf = clip_range_vf

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if log and parent_rng is None:
            self.setup_wandb(self.project_name, self.experiment_name, wandb_entity)

    def __deepcopy__(self, memo):
        """Deep copy the policy."""
        copied_net = deepcopy(self.net)
        copied = type(self)(
            self.env,
            self.scalarization,
            self.weights,
            self.id,
            self.buffer_size,
            self.net_arch,
            self.gamma,
            self.learning_rate,
            self.project_name,
            self.experiment_name,
            log=self.log,
            device=self.device,
            parent_rng=self.parent_rng,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_policy_net(self) -> nn.Module:
        return self.net

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        raise Exception("On-policy algorithms should not share buffer.")

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    @th.no_grad()
    @override
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).to(self.device)
        else:
            obs = th.as_tensor(obs).to(self.device)
        accrued_reward = th.as_tensor(accrued_reward).float().to(self.device)
        return self.__choose_action(obs, accrued_reward)[0]

    @th.no_grad()
    def __choose_action(self, obs: th.Tensor, accrued_reward: th.Tensor) -> tuple[int, np.float32]:
        distribution = self.net.distribution(obs, accrued_reward)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.detach().item(), log_prob.detach().item()
    
    def evaluate_actions(self, obs: th.Tensor, accrued_rewards: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        """
        distribution = self.net.distribution(obs, accrued_rewards)
        values = self.net.get_value(obs, accrued_rewards)
        log_probs = distribution.log_prob(actions.squeeze())
        return values, log_probs, distribution.entropy()

    
    @th.no_grad()
    def get_action_and_value(self, obs: th.Tensor, accrued_reward: th.Tensor) -> tuple[int, np.float32, np.float32]:
        action, log_prob = self.__choose_action(obs, accrued_reward)
        value = self.net.get_value(obs, accrued_reward)
        return action, log_prob, value

    @override
    def update(self):
        (
            obs,
            accrued_rewards,
            actions,
            rewards,
            old_log_probs,
            old_state_values,
            next_obs,
            terminateds,
        ) = self.buffer.get_all_data(to_tensor=True, device=self.device)
        clip_range = self.clip_range(self.global_step)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self.global_step)

        episodic_return = th.sum(rewards, dim=0)
        scalarized_return = self.scalarization(episodic_return.cpu().numpy(), self.weights)
        scalarized_return = th.scalar_tensor(scalarized_return).to(self.device)

        discounted_forward_rewards = self._forward_cumulative_rewards(rewards)
        scalarized_action_values = self.scalarization(discounted_forward_rewards)
        # TODO FdH: should we do discounting here? -- no this is probably part of the estimation
        state_values, log_probs, entropy = self.evaluate_actions(obs, accrued_rewards, actions)
        # print(log_probs)
        # print(state_values)
        scalarized_state_values = self.scalarization(state_values)
        # For each sample in the batch, get the distribution over actions
        # current_distribution = self.net.distribution(obs, accrued_rewards)
        # Policy gradient
        # TODO FdH: change to PPO loss
        ratio = th.exp(log_probs - old_log_probs)
        # clipped surrogate loss
        if self.use_advantage:
            advantages = self.get_advantages(scalarized_action_values, state_values)
        else:
            advantages = scalarized_action_values
        if self.ppo_clip:
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        else:
            policy_loss = -th.mean(log_probs * advantages)

        if self.clip_range_vf is None:
            # No clipping
            values_pred = state_values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_state_values + th.clamp(
                state_values - old_state_values, -clip_range_vf, clip_range_vf
            )
        value_loss = F.mse_loss(scalarized_return, values_pred)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_probs)
        else:
            entropy_loss = -th.mean(entropy)
        # PPO loss
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        # EUPG loss
        # print(self.scalarization)
        # print(log_probs, scalarized_state_values)
        # loss = -th.mean(log_probs * scalarized_action_values)

        #TODO FdH: investigate early stopping based on KL divergence (see SB3 implementation)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            wandb.log(
                {
                    f"losses{log_str}/policy_loss": policy_loss,
                    f"losses{log_str}/entropy_loss": entropy_loss,
                    f"losses{log_scr}/value_loss": value_loss,
                    f"losses{log_str}/loss": loss,
                    f"metrics{log_str}/scalarized_state_values": scalarized_state_values.mean(),
                    f"metrics{log_str}/scalarized_episodic_return": scalarized_return,
                    "global_step": self.global_step,
                },
            )

    def _forward_cumulative_rewards(self, rewards):
        flip_rewards = rewards.flip(dims=[0])
        cumulative_rewards = th.zeros(self.reward_dim).to(self.device)
        for i in range(len(rewards)):
            cumulative_rewards = self.gamma * cumulative_rewards + flip_rewards[i]
            flip_rewards[i] = cumulative_rewards
        forward_rewards = flip_rewards.flip(dims=[0])
        return forward_rewards
    
    def get_advantages(self, returns, values):
        # TODO FdH: this should be simply the Monte-Carlo advantage estimator, validate this
        return returns - values


    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, eval_freq: int = 1000, start_time=None):
        """Train the agent.

        Args:
            total_timesteps: Number of timesteps to train for
            eval_env: Environment to run policy evaluation on
            eval_freq: Frequency of policy evaluation
            start_time: Start time of the training (for SPS)
        """
        if start_time is None:
            start_time = time.time()
        # Init
        (
            obs,
            _,
        ) = self.env.reset()
        accrued_reward_tensor = th.zeros(self.reward_dim, dtype=th.float32).float().to(self.device)

        # Training loop
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            with th.no_grad():
                # For training, takes action according to the policy
                action, log_prob, value = self.get_action_and_value(th.Tensor([obs]).to(self.device), accrued_reward_tensor)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            value = 0.0
            # Memory update
            self.buffer.add(obs, accrued_reward_tensor.cpu().numpy(), action, vec_reward, next_obs, log_prob, value, terminated)
            accrued_reward_tensor += th.from_numpy(vec_reward).to(self.device)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval_esr(eval_env, scalarization=self.scalarization, weights=self.weights, log=self.log)

            if terminated or truncated:
                # NN is updated at the end of each episode
                self.update()
                self.buffer.cleanup()
                obs, _ = self.env.reset()
                self.num_episodes += 1
                accrued_reward_tensor = th.zeros(self.reward_dim).float().to(self.device)

                if self.log and self.num_episodes % self.log_every == 0 and "episode" in info.keys():
                    log_episode_info(
                        info=info["episode"],
                        scalarization=self.scalarization,
                        weights=self.weights,
                        id=self.id,
                        global_timestep=self.global_step,
                    )

            else:
                obs = next_obs

            if self.log and self.global_step % 1000 == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                wandb.log({"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step})

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "seed": self.seed,
        }

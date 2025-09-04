"""Accrued reward buffer for ESR algorithms."""

import numpy as np
import torch as th
from typing import Callable


class ValueProbAccruedRewardReplayBuffer:
    """Replay buffer with log-probabilities, value-estimates, and accrued rewards stored (for ESR algorithms)."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        """Initialize the Replay Buffer.

        Args:
            obs_shape: Shape of the observations
            action_shape:  Shape of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.rew_dim = rew_dim
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype
        self.cleanup()
    

    def add(self, obs, accrued_reward, action, reward, next_obs, state_value, log_prob, done):
        """Add a new experience to memory.

        Args:
            obs: Observation
            accrued_reward: Accrued reward
            action: Action
            reward: Reward
            next_obs: Next observation
            log_prob: the log-probability of `action`
            value: the vectorial value estimate of `obs`
            done: Done
        """
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        # TODO FdH: ensure that the right kind of deep copy is made here for log_probs, values
        self.state_values[self.ptr] = state_value.detach().clone()
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def discounted_cumulative_rewards(self, rewards, gamma, device):
        flip_rewards = th.tensor(rewards).flip(dims=[0])
        cumulative_rewards = th.zeros(rewards.shape[1]).to(device)
        # print("shape of rewards", rewards.shape[1])
        for i in range(len(flip_rewards)):
            cumulative_rewards = gamma * cumulative_rewards + flip_rewards[i]
            flip_rewards[i] = cumulative_rewards
        forward_rewards = flip_rewards.flip(dims=[0])
        return forward_rewards
    
    def compute_returns_and_advantages(self, scalarization: Callable[[np.ndarray, np.ndarray], float], gamma, device) -> None:
        """
        Per-episode post-processing that computes the returns for the episode currently in the buffer.
        Assumes that self.ptr points at next index after end of episode.
        """
        #assert self.dones.sum() == 1.0, f"Currently has {self.dones.sum()} episodes in the buffer"
        episode_start = 0 # assume that there is only 1 episode in the buffer
        episode_end = self.size - 1 # assume that episode ends at previous time step
        assert self.dones[episode_end] == True
        inds = np.arange(self.size)
        discounted_forward_return = self.discounted_cumulative_rewards(self.rewards[inds], gamma, device)
        advantages = discounted_forward_return - self.state_values[inds] # MC estimate
        self.scal_advantages = scalarization(advantages) # MC estimate
        self.returns = discounted_forward_return

    
    def __sample(self, inds, to_tensor=False, device=None):
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.log_probs[inds],
            self.state_values[inds],
            self.returns[inds],
            self.scal_advantages[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences.

        Args:
            batch_size: Number of elements to sample
            replace: Whether to sample with replacement or not
            use_cer: Whether to use CER or not
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, log_probs, values dones)
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        return self.__sample(inds, to_tensor, device)

    def cleanup(self):
        """Cleanup the buffer."""
        self.size, self.ptr = 0, 0
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((self.max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_obs = np.zeros((self.max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.actions = np.zeros((self.max_size,) + self.action_shape, dtype=self.action_dtype)
        self.rewards = np.zeros((self.max_size, self.rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((self.max_size, self.rew_dim), dtype=np.float32)
        self.state_values = np.zeros((self.max_size, self.rew_dim), dtype=np.float32)
        self.returns = np.zeros((self.max_size, self.rew_dim), dtype=np.float32)
        self.scal_advantages = np.zeros((self.max_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.max_size,),dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def get_all_data(self, to_tensor=False, device=None):
        """Returns the whole buffer.

        Args:
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, log_probs, values, dones)
        """
        inds = np.arange(self.size)
        return self.__sample(inds, to_tensor, device)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size

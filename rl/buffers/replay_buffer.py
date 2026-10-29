"""Uniform experience replay buffer for off-policy algorithms (DQN, DDPG, SAC, ...)."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size circular buffer storing transitions as numpy arrays.

    Args:
        capacity: Maximum number of transitions kept; oldest are overwritten.
        obs_shape: Shape of a single observation.
        action_shape: Shape of a single action (``()`` for discrete actions).
        action_dtype: ``np.int64`` for discrete, ``np.float32`` for continuous.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        action_dtype: type = np.int64,
    ) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> Batch:
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from a buffer of size {self._size}")
        idx = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x[idx], device=device)

        return Batch(
            obs=to_tensor(self.obs),
            actions=to_tensor(self.actions),
            rewards=to_tensor(self.rewards),
            next_obs=to_tensor(self.next_obs),
            dones=to_tensor(self.dones),
        )

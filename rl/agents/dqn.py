"""Deep Q-Network (Mnih et al., 2015) with a target network and optional Double DQN update."""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from rl.buffers import Batch
from rl.networks import MLP


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    target_update_interval: int = 500  # in gradient steps
    double_dqn: bool = True
    max_grad_norm: float = 10.0


class DQNAgent:
    """Value-based agent for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.n_actions = n_actions
        self.device = torch.device(device)

        self.q_net = MLP(obs_dim, n_actions, config.hidden_sizes).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q_net(obs_t).argmax(dim=1).item())

    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step on the TD error. Returns training metrics."""
        q = self.q_net(batch.obs).gather(1, batch.actions.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Online net chooses the action, target net evaluates it.
                next_actions = self.q_net(batch.next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(batch.next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(batch.next_obs).max(dim=1).values
            target = batch.rewards + self.cfg.gamma * (1.0 - batch.dones) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "q_mean": q.mean().item()}

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)

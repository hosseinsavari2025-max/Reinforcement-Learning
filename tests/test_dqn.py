import numpy as np

from rl.agents import DQNAgent, DQNConfig
from rl.buffers import ReplayBuffer


def test_act_returns_valid_action():
    agent = DQNAgent(obs_dim=4, n_actions=3, config=DQNConfig())
    for eps in (0.0, 1.0):
        assert 0 <= agent.act(np.zeros(4, dtype=np.float32), epsilon=eps) < 3


def test_update_runs_and_syncs_target():
    agent = DQNAgent(obs_dim=4, n_actions=2, config=DQNConfig(target_update_interval=2))
    buf = ReplayBuffer(capacity=64, obs_shape=(4,))
    for _ in range(64):
        buf.add(np.random.randn(4), np.random.randint(2), 1.0, np.random.randn(4), done=False)

    metrics = agent.update(buf.sample(32))
    assert np.isfinite(metrics["loss"])
    agent.update(buf.sample(32))  # second update triggers target sync
    for p, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert (p == tp).all()

import numpy as np
import pytest

from rl.buffers import ReplayBuffer


def test_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_shape=(4,))
    for i in range(10):
        buf.add(np.full(4, i), i % 2, 1.0, np.full(4, i + 1), done=False)
    batch = buf.sample(8)
    assert len(buf) == 10
    assert batch.obs.shape == (8, 4)
    assert batch.actions.shape == (8,)
    assert batch.rewards.shape == (8,)


def test_circular_overwrite():
    buf = ReplayBuffer(capacity=5, obs_shape=(1,))
    for i in range(12):
        buf.add([i], 0, 0.0, [i], done=False)
    assert len(buf) == 5
    assert set(buf.obs[:, 0].astype(int)) == {7, 8, 9, 10, 11}


def test_sample_too_many_raises():
    buf = ReplayBuffer(capacity=10, obs_shape=(2,))
    with pytest.raises(ValueError):
        buf.sample(1)

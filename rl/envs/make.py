"""Environment construction. Custom wrappers and environments live in this package."""

import gymnasium as gym


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create a Gymnasium environment with episode statistics recorded in ``info['episode']``."""
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env

"""Evaluate a trained agent.

Usage:
    python scripts/evaluate.py --run-dir runs/dqn_cartpole_<timestamp> --episodes 10
    python scripts/evaluate.py --run-dir runs/dqn_cartpole_<timestamp> --render   # watch it play
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from rl.agents import DQNAgent, DQNConfig
from rl.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    env = make_env(cfg["env_id"], seed=cfg["seed"] + 1000, render_mode="human" if args.render else None)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, DQNConfig(**cfg["agent"]))
    agent.load(str(run_dir / "model.pt"))

    returns = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, epsilon=0.0))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    print(f"{cfg['env_id']}: mean return {np.mean(returns):.1f} ± {np.std(returns):.1f} over {args.episodes} episodes")


if __name__ == "__main__":
    main()

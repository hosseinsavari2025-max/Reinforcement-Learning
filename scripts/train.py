"""Train an agent from a YAML config.

Usage:
    python scripts/train.py --config configs/dqn_cartpole.yaml
    python scripts/train.py --config configs/dqn_cartpole.yaml --total-steps 5000   # quick smoke test
"""

import argparse
import time
from pathlib import Path

import yaml

from rl.agents import DQNAgent, DQNConfig
from rl.buffers import ReplayBuffer
from rl.envs import make_env
from rl.utils import CSVLogger, linear_schedule, set_seed
from rl.utils.plotting import plot_learning_curve


def train_dqn(cfg: dict, run_dir: Path) -> DQNAgent:
    set_seed(cfg["seed"])
    env = make_env(cfg["env_id"], seed=cfg["seed"])
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, DQNConfig(**cfg["agent"]))
    buffer = ReplayBuffer(cfg["buffer_size"], env.observation_space.shape)
    logger = CSVLogger(run_dir / "progress.csv", ["step", "episode_return", "episode_length", "epsilon"])

    obs, _ = env.reset(seed=cfg["seed"])
    start = time.time()
    episodes = 0
    for step in range(1, cfg["total_steps"] + 1):
        epsilon = linear_schedule(cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_steps"], step)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Bootstrap through time-limit truncation; only true terminations end the return.
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            episodes += 1
            ep = info["episode"]
            logger.log(step=step, episode_return=float(ep["r"]), episode_length=int(ep["l"]), epsilon=round(epsilon, 3))
            if episodes % 20 == 0:
                print(f"step {step:>7} | episode {episodes:>4} | return {float(ep['r']):6.1f} | eps {epsilon:.2f} | {time.time() - start:.0f}s")
            obs, _ = env.reset()

        if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            agent.update(buffer.sample(cfg["batch_size"]))

    logger.close()
    env.close()
    agent.save(str(run_dir / "model.pt"))
    plot_learning_curve(run_dir / "progress.csv", run_dir / "learning_curve.png", title=f"DQN — {cfg['env_id']}")
    return agent


TRAINERS = {"dqn": train_dqn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-steps", type=int, default=None, help="override total_steps from the config")
    parser.add_argument("--run-dir", default=None, help="defaults to runs/<config-name>_<timestamp>")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
        cfg["learning_starts"] = min(cfg["learning_starts"], args.total_steps // 2)

    run_dir = Path(args.run_dir or f"runs/{Path(args.config).stem}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    TRAINERS[cfg["algo"]](cfg, run_dir)
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()

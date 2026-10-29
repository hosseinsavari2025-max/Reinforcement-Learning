# Reinforcement Learning

Reinforcement learning algorithms implemented from scratch in PyTorch, alongside lesson notes and reproducible experiments.

## Repository structure

```
Reinforcement-Learning/
├── rl/                 # reusable Python package
│   ├── agents/         # DQN (more to come: REINFORCE, PPO, SAC)
│   ├── networks/       # MLP and other function approximators
│   ├── buffers/        # replay / rollout buffers
│   ├── envs/           # env construction, wrappers, custom envs
│   └── utils/          # seeding, schedules, logging, plotting
├── lessons/            # learning path: 01_bandits → 07_continuous_control
├── configs/            # one YAML file per experiment
├── scripts/            # train.py, evaluate.py
├── results/            # curated plots and tables
├── tests/              # unit tests (pytest)
└── docs/               # theory notes and paper summaries
```

## Quick start

```bash
git clone https://github.com/hosseinsavari2025-max/Reinforcement-Learning.git
cd Reinforcement-Learning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# train DQN on CartPole (a few minutes on CPU)
python scripts/train.py --config configs/dqn_cartpole.yaml

# evaluate / watch the trained agent
python scripts/evaluate.py --run-dir runs/<run-name> --render

# run tests
pytest
```

Each run writes `config.yaml`, `progress.csv`, `model.pt` and `learning_curve.png` to `runs/<run-name>/`.

## Roadmap

| Lesson | Algorithm | Environment | Status |
|--------|-----------|-------------|--------|
| 01 | ε-greedy, UCB, Thompson | k-armed bandit | ⬜ |
| 02 | Policy / value iteration | FrozenLake | ⬜ |
| 03 | SARSA, Q-learning | CliffWalking | ⬜ |
| 04 | DQN, Double DQN | CartPole | ✅ implemented |
| 05 | REINFORCE | CartPole | ⬜ |
| 06 | A2C, PPO | LunarLander | ⬜ |
| 07 | DDPG, TD3, SAC | Pendulum, MuJoCo | ⬜ |

## Results

| Algorithm | Environment | Mean return | Steps |
|-----------|-------------|-------------|-------|
| DQN | CartPole-v1 | — | — |

## Related projects

- [wildfire-rl](https://github.com/hosseinsavari2025-max/Competition/tree/main/wildfire-rl) — RL for wildfire simulation

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- OpenAI Spinning Up
- Mnih et al., *Human-level control through deep reinforcement learning*, 2015

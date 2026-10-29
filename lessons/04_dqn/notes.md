# 04 — Deep Q-Networks

**Goal:** scale Q-learning with a neural network.

Implementation: [`rl/agents/dqn.py`](../../rl/agents/dqn.py)

Key ingredients:
- **Experience replay** — break correlation between consecutive samples
- **Target network** — stabilise the bootstrapped target
- **Double DQN** — online net selects `a'`, target net evaluates it, reducing overestimation
- Huber loss and gradient clipping

Run it:
```bash
python scripts/train.py --config configs/dqn_cartpole.yaml
python scripts/evaluate.py --run-dir runs/<your-run>
```

**Next:** Dueling architecture, prioritized replay, LunarLander.

**Reading:** Mnih et al. 2015; van Hasselt et al. 2016.

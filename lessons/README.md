# Lessons

A step-by-step path through RL, roughly following Sutton & Barto and Spinning Up.
Each folder holds `notes.md` (theory summary + key equations) and one or more notebooks.
Notebooks import reusable code from the `rl/` package; anything worth reusing gets moved there.

| # | Topic | Algorithms | Environment |
|---|-------|-----------|-------------|
| 01 | Multi-armed bandits | ε-greedy, UCB, Thompson sampling | k-armed testbed |
| 02 | Dynamic programming | Policy evaluation, policy/value iteration | FrozenLake |
| 03 | Monte Carlo & TD | MC control, SARSA, Q-learning | CliffWalking, Taxi |
| 04 | Deep Q-learning | DQN, Double DQN, Dueling | CartPole, LunarLander |
| 05 | Policy gradient | REINFORCE (+ baseline) | CartPole |
| 06 | Actor-critic | A2C, PPO | LunarLander |
| 07 | Continuous control | DDPG, TD3, SAC | Pendulum, MuJoCo |

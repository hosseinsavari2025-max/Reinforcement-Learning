# 02 — Dynamic programming

**Goal:** solve a known MDP exactly.

- Bellman expectation: `V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γV(s')]`
- Bellman optimality: `V*(s) = max_a Σ_{s',r} p(s',r|s,a) [r + γV*(s')]`
- Policy iteration = evaluation + greedy improvement until stable
- Value iteration = repeated optimality backups

**Exercise:** value iteration on `FrozenLake-v1` (use `env.unwrapped.P` for the model).

**Reading:** Sutton & Barto, Chapter 4.

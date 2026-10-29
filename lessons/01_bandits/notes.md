# 01 — Multi-armed bandits

**Goal:** understand the exploration–exploitation trade-off without states.

- Action-value estimate: `Q(a) ← Q(a) + (1/N(a)) · (R − Q(a))`
- ε-greedy: random action with probability ε, otherwise `argmax Q(a)`
- UCB: `argmax [Q(a) + c·sqrt(ln t / N(a))]`
- Thompson sampling: sample from the posterior of each arm, pick the best sample

**Exercise:** reproduce Sutton & Barto Figure 2.2 (10-armed testbed, ε ∈ {0, 0.01, 0.1}).

**Reading:** Sutton & Barto, Chapter 2.

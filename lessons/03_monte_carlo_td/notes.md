# 03 — Monte Carlo and temporal-difference learning

**Goal:** learn without a model, from sampled experience.

- MC: update toward the full return `G_t`
- SARSA (on-policy): `Q(s,a) ← Q(s,a) + α [r + γQ(s',a') − Q(s,a)]`
- Q-learning (off-policy): `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`

**Exercise:** compare SARSA vs Q-learning on `CliffWalking-v0` (Sutton & Barto Example 6.6).

**Reading:** Sutton & Barto, Chapters 5–6.

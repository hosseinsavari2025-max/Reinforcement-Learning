# 05 — Policy gradient

**Goal:** optimise the policy directly.

- Policy gradient theorem: `∇J(θ) = E[ Σ_t ∇log π_θ(a_t|s_t) · G_t ]`
- Baseline `b(s)` reduces variance without adding bias: replace `G_t` with `G_t − b(s_t)`

**Exercise:** REINFORCE with and without a baseline on CartPole; compare variance across 5 seeds.

**Reading:** Sutton & Barto Ch. 13; Williams 1992.

# 06 — Actor-critic: A2C and PPO

**Goal:** combine a learned critic with a policy gradient.

- Advantage: `A(s,a) ≈ r + γV(s') − V(s)`; use GAE(λ) in practice
- PPO clipped objective: `E[min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)]`, `ρ = π_θ / π_old`

**Exercise:** PPO on `LunarLander-v3`, target return ≥ 200.

**Reading:** Schulman et al. 2017 (PPO); Schulman et al. 2016 (GAE); "The 37 implementation details of PPO".

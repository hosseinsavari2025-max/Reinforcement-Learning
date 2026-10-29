# 07 — Continuous control

**Goal:** off-policy actor-critic for continuous actions.

- DDPG: deterministic actor + Q critic
- TD3: twin critics, delayed actor updates, target policy smoothing
- SAC: maximum-entropy objective with automatic temperature tuning

**Exercise:** SAC on `Pendulum-v1`, then a MuJoCo task (`HalfCheetah-v5`).

**Reading:** Lillicrap et al. 2016; Fujimoto et al. 2018; Haarnoja et al. 2018.

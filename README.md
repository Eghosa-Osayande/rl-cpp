# Reinforcement Learning for Coverage Path Planning 

```(Presented as a Poster at the 2026 Towards Autonomous Robotic Systems Conference held at the University of Manchester.)```

A comparison of **planning-oriented** and **control-oriented** action spaces for deep reinforcement learning-based coverage path planning (CPP) in unknown environments.

This project extends the approach of [Jonnarth et al. (2024)](https://proceedings.mlr.press/v235/jonnarth24a.html), replacing low-level velocity commands with higher-level direction-and-reach motion primitives.

## Overview

The framework uses **Soft Actor-Critic (SAC)** and a **Scale-Grouped Convolutional Neural Network (SGCNN)** to learn coverage policies from multiscale egocentric coverage, obstacle, and frontier maps, alongside ranging measurements.

Two action spaces are compared under the same learning framework and reward function:

- **Control-oriented:** \(a=(v,\omega)\) — predicts linear and angular velocities at each policy step.
- **Planning-oriented:** \(a=(\Delta\theta,r)\) — predicts a relative heading change and travel distance, executed over multiple control cycles.

## Experiments

Policies were trained and evaluated using the Tier 0–2 environments of [Explore-Bench](https://doi.org/10.1109/ICRA46639.2022.9812164) in an omnidirectional exploration setting.

| Parameter | Value |
|---|---|
| Algorithm | SAC |
| Feature extractor | SGCNN |
| Training iterations | 400,000 |
| Learning rate | 1e-5 |
| Batch size | 256 |
| Discount factor | 0.99 |
| GPU | NVIDIA T4 |

## Results

Mean ± standard deviation:

| Metric | Control (v, ω) | Planning (Δθ, r) |
|---|---:|---:|
| Coverage | 0.84 ± 0.10 | 0.80 ± 0.18 |
| Collision rate | 0.33 ± 0.58 | 0.42 ± 0.51 |
| Overlap (%) | 0.54 ± 0.24 | 3.82 ± 8.38 |
| Policy decisions | 1753.83 ± 672.11 | 346.75 ± 177.35 |

**Key findings:**

- Planning-oriented actions require approximately **5× fewer policy decisions**.
- Both formulations achieve a median coverage of 0.90 and median collision rate of zero.
- Planning-oriented actions exhibit higher overlap and variability, indicating less consistent path efficiency.

These results suggest that higher-level motion primitives can substantially reduce policy intervention frequency while maintaining broadly comparable coverage and collision performance.

## Future Work

- Extend training to 8 million iterations.
- Evaluate generalisation on Explore-Bench Tiers 3–5.
- Improve the path efficiency of planning-oriented actions.

## References

1. Jonnarth, A., Zhao, J., & Felsberg, M. (2024). *Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning*. ICML.
2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
3. Xu, Y., et al. (2022). *Explore-Bench: Data Sets, Metrics and Evaluations for Frontier-Based and Deep-Reinforcement-Learning-Based Autonomous Exploration*. ICRA.

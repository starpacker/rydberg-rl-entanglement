# 附录 F：PPO 伪代码 + 超参数表

## F.1 引言

Proximal Policy Optimization (PPO) [PPO17] 是一种 on-policy reinforcement learning 算法，通过 clipped surrogate objective 实现稳定的策略更新。本附录给出 PPO 应用于 Rydberg quantum control 的完整伪代码和超参数配置。

## F.2 算法伪代码

```
Algorithm: PPO for Rydberg Quantum Control
────────────────────────────────────────────────────
Input: Initial policy π_θ, value function V_θ, environment E
       Hyperparameters: ε, γ, λ, K, c₁, c₂, n_steps, batch_size

 1. for iteration = 1, 2, ... do
 2.   // ── 数据采集阶段 ──
 3.   Collect n_steps transitions {(s_t, a_t, r_t, s_{t+1})} by running π_θ in E
 4.     (on each episode reset, apply domain randomization to physical parameters)
 5.   Compute returns R_t = Σ_{k=0}^{T-t} γ^k r_{t+k}
 6.   Compute advantages Â_t using GAE(γ, λ):
 7.     δ_t = r_t + γ V_θ(s_{t+1}) - V_θ(s_t)
 8.     Â_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}
 9.   Normalize advantages: Â_t ← (Â_t - mean(Â)) / (std(Â) + 1e-8)
10.   Store π_θ_old(a_t|s_t) for all collected transitions
11.
12.   // ── 策略更新阶段 ──
13.   for epoch = 1, ..., K do
14.     Shuffle collected data
15.     for minibatch B ⊂ collected transitions do
16.       // Importance sampling ratio
17.       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
18.
19.       // Clipped surrogate objective
20.       L_CLIP = (1/|B|) Σ_{t∈B} min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)
21.
22.       // Value function loss
23.       L_VF = (1/|B|) Σ_{t∈B} (V_θ(s_t) - R_t)²
24.
25.       // Entropy bonus
26.       H = (1/|B|) Σ_{t∈B} H(π_θ(·|s_t))
27.
28.       // Total loss (gradient ascent on L)
29.       L = L_CLIP - c₁·L_VF + c₂·H
30.
31.       Update θ via Adam optimizer: θ ← θ + α·∇_θ L
32.     end for
33.   end for
34. end for
────────────────────────────────────────────────────
Output: Optimized policy π_θ*
```

**关键设计决策说明**：

- **Line 4: Domain randomization**：每次 episode reset 时，物理参数（如 $C_6$、$\Omega_{\max}$、decoherence rates）在合理范围内随机采样。这使得 learned policy 对参数不确定性具有 robustness。
- **Line 6-8: GAE($\gamma$, $\lambda$)**：Generalized Advantage Estimation 通过 $\lambda$ 参数在 bias 和 variance 之间插值。$\lambda = 1$ 等价于 Monte Carlo estimate（低 bias、高 variance），$\lambda = 0$ 等价于 TD(0) estimate（高 bias、低 variance）。
- **Line 9: Advantage normalization**：标准化 advantages 提高训练稳定性，避免不同 episode 之间 reward scale 的差异影响梯度更新。
- **Line 20: Clipping**：$\epsilon$ 限制了每次更新中策略变化的幅度，防止 catastrophic policy updates。

## F.3 超参数表

| 超参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| Learning rate | $\alpha$ | $3\times10^{-4}$ | Adam optimizer 学习率 |
| Steps per rollout | $n_{\text{steps}}$ | 2048 | 每次采集的 transition 数 |
| Minibatch size | $|B|$ | 64 | 小批量大小 |
| Epochs per update | $K$ | 10 | 每次数据采集后的 epoch 数 |
| Discount factor | $\gamma$ | 1.0 | 无折扣（关心终态 fidelity） |
| GAE parameter | $\lambda$ | 0.95 | Advantage estimation 平滑参数 |
| Clip range | $\varepsilon$ | 0.2 | PPO 裁剪范围 |
| Value loss coefficient | $c_1$ | 0.5 | Value function loss 权重 |
| Entropy coefficient | $c_2$ | 0.01 | 熵正则化系数，鼓励 exploration |
| Total timesteps | — | 50,000 | 总训练步数 |
| Network architecture | — | MLP [64, 64] | Actor-Critic 共享网络，两层 fully-connected |
| Activation function | — | Tanh | 隐藏层激活函数 |
| Max grad norm | — | 0.5 | Gradient clipping 阈值 |

**超参数选择依据**：

- **$\gamma = 1.0$**：由于 Rydberg gate 的 reward 仅在 episode 结束时给出（terminal fidelity），无需时间折扣。所有时间步的 action 对最终 fidelity 同等重要。
- **$n_{\text{steps}} = 2048$**：对于 episode length = 30 步的环境，2048 步约包含 68 个 complete episodes，提供足够的统计量。
- **$c_2 = 0.01$**：较小但非零的 entropy coefficient 防止策略过早收敛到确定性策略，保持一定的 exploration 能力。

## F.4 环境参数

| 参数 | 值 | 说明 |
|---|---|---|
| State dimension | 32 | 密度矩阵 $\rho$ 的 real + imaginary 部分展平 |
| Action dimension | 2 | 归一化的 $\Omega(t)$ 和 $\Delta(t)$ |
| Episode length | 30 steps | 对应 gate time $T$ 的等分 |
| Time step | $\Delta t = T/30$ | 每步控制参数恒定 |
| Reward | $r_T = \mathcal{F}(\rho_{\text{final}}, \rho_{\text{target}})$ | 终态 fidelity |

**State representation**：4×4 密度矩阵 $\rho$ 具有 16 个复数元素，拆分为实部和虚部后为 32 维实向量。由于 $\rho$ 为 Hermitian，实际独立参数更少（$d^2 - 1 = 15$），但使用完整 32 维表示简化了实现且不影响 learning 效率。

**Action space**：连续动作空间 $\mathbf{a} = (a_\Omega, a_\Delta) \in [-1, 1]^2$，通过线性映射转换为物理参数：

$$\Omega(t) = \frac{a_\Omega + 1}{2} \cdot \Omega_{\max}, \quad \Delta(t) = a_\Delta \cdot \Delta_{\max}$$

其中 $\Omega_{\max}$ 和 $\Delta_{\max}$ 为实验可达范围的上限。

## F.5 小结

本附录给出的 PPO 实现遵循 Stable Baselines3 的标准接口 [PPO17]。超参数设置基于对 Rydberg quantum control 环境特点的分析（short episodes、terminal reward、continuous actions），部分参数经过 grid search 微调。Environment 的 domain randomization 设计使得 trained policy 可直接迁移到参数不确定的实际实验场景。

---

**参考文献**

- [PPO17] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).

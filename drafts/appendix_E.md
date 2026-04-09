# 附录 E：Lindblad 数值方法

## E.1 引言

Rydberg quantum gate 的 realistic simulation 必须考虑 decoherence effects，包括 spontaneous emission、dephasing 和 finite temperature 效应。Lindblad master equation 提供了描述 Markovian open quantum systems 的标准框架。本附录介绍 Lindblad 方程的数学形式、Kraus representation 的推导，以及两种主要数值求解方法（mesolve 和 mcsolve）的比较。

## E.2 Lindblad Master Equation

在 Markov approximation 下，系统密度矩阵 $\rho$ 的演化由 Lindblad master equation 描述：

$$\dot{\rho} = -i[H,\rho] + \sum_k \gamma_k\left(L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right) \tag{E.1}$$

其中：

- $H$ 为系统 Hamiltonian（可以是 time-dependent 的）。
- $L_k$ 为第 $k$ 个 Lindblad（collapse/jump）operator，描述特定的 dissipation channel。
- $\gamma_k$ 为对应的 dissipation rate。
- $\{A, B\} = AB + BA$ 为 anticommutator。

对于 Rydberg atom 系统，典型的 Lindblad operators 包括：

| Dissipation channel | $L_k$ | $\gamma_k$ |
|---|---|---|
| Rydberg state 自发辐射 | $\|g\rangle\langle r\|$ | $\Gamma_r \sim 1/(n^3\tau_0)$ |
| Intermediate state 自发辐射 | $\|g\rangle\langle e\|$ | $\gamma_e \approx 2\pi \times 6.07$ MHz (Rb $5P_{3/2}$) |
| Laser phase noise (dephasing) | $\|r\rangle\langle r\| - \|g\rangle\langle g\|$ | $\gamma_\phi$ |

## E.3 Kraus Representation

对于一个短时间步 $\delta t$，Lindblad 演化可以近似为 Kraus map：

$$\rho(t+\delta t) \approx \sum_\mu K_\mu \rho(t) K_\mu^\dagger \tag{E.2}$$

**推导**：将式 (E.1) 离散化，$\rho(t+\delta t) \approx \rho(t) + \dot{\rho}\,\delta t$，整理得到：

$$\rho(t+\delta t) \approx \left(I - iH\delta t - \frac{1}{2}\sum_k \gamma_k L_k^\dagger L_k\,\delta t\right)\rho\left(I + iH\delta t - \frac{1}{2}\sum_k \gamma_k L_k^\dagger L_k\,\delta t\right) + \sum_k \gamma_k\,\delta t\, L_k\rho L_k^\dagger + O(\delta t^2)$$

由此识别出 Kraus operators：

- **No-jump operator**（非跳跃演化）：

$$K_0 = I - \left(iH + \frac{1}{2}\sum_k \gamma_k L_k^\dagger L_k\right)\delta t$$

这描述了系统在没有 quantum jump 发生时的（非幺正）演化。$K_0$ 不是 unitary 的，其效果是使 state 的 norm 缓慢减小，反映 jump 事件可能已发生的概率。

- **Jump operators**（跳跃算符），对每个 $k$：

$$K_k = \sqrt{\gamma_k\,\delta t}\, L_k$$

每个 $K_k$ 对应在时间步 $\delta t$ 内发生第 $k$ 种 dissipative event（如 photon emission）的振幅。

可以验证 trace-preserving condition $\sum_\mu K_\mu^\dagger K_\mu = I + O(\delta t^2)$ 在一阶近似下成立。

## E.4 数值求解方法：mesolve vs mcsolve

### E.4.1 mesolve（密度矩阵直接积分）

**方法**：将式 (E.1) 视为关于 $\rho$ 的 ordinary differential equation，直接数值积分。

**实现**：将 $d \times d$ 的密度矩阵 $\rho$ 展开为 $d^2$ 维向量 $\text{vec}(\rho)$。Lindblad 方程可写为：

$$\frac{d}{dt}\text{vec}(\rho) = \mathcal{L}\,\text{vec}(\rho)$$

其中 $\mathcal{L}$ 为 $d^2 \times d^2$ 的 Liouvillian superoperator。

**优势**：

- 确定性方法，单次运行即给出精确结果（在 Markov 近似下）。
- 自动处理所有 decoherence channels 的相干叠加效应。

**劣势**：

- 计算复杂度为 $O(d^4)$ per time step（需要 $d^2 \times d^2$ 矩阵运算）。
- 内存需求为 $O(d^4)$。

### E.4.2 mcsolve（Monte Carlo Wavefunction）

**方法**：不直接演化密度矩阵，而是模拟单个 stochastic quantum trajectory。每条 trajectory 演化一个纯态 $|\psi(t)\rangle$，具有以下规则：

1. **Smooth evolution**：用 non-Hermitian Hamiltonian $H_{\text{eff}} = H - \frac{i}{2}\sum_k \gamma_k L_k^\dagger L_k$ 演化态矢。态矢的 norm 会逐渐减小。

2. **Quantum jump**：在每个时间步，以概率 $\delta p = 1 - \langle\psi|e^{iH_{\text{eff}}^\dagger\delta t}e^{-iH_{\text{eff}}\delta t}|\psi\rangle \approx \delta t \sum_k \gamma_k\langle\psi|L_k^\dagger L_k|\psi\rangle$ 发生跳跃。若跳跃发生，态矢按 $|\psi\rangle \to L_k|\psi\rangle/\|L_k|\psi\rangle\|$ 更新（选择哪个 $L_k$ 的概率正比于 $\gamma_k\langle\psi|L_k^\dagger L_k|\psi\rangle$）。

3. **Averaging**：对 $M$ 条 trajectories 取平均恢复密度矩阵：$\rho(t) \approx \frac{1}{M}\sum_{i=1}^{M} |\psi^{(i)}(t)\rangle\langle\psi^{(i)}(t)|$。

**优势**：

- 每条 trajectory 的计算复杂度为 $O(d^2)$（仅需 matrix-vector 乘法）。
- 内存需求为 $O(d^2)$（仅存储一个 state vector）。
- 对于大 Hilbert space（$d \gg 1$），当所需 trajectory 数 $M \ll d^2$ 时，总成本 $O(Md^2)$ 远小于 mesolve 的 $O(d^4)$。

**劣势**：

- 结果具有 statistical noise，精度随 $1/\sqrt{M}$ 提高。
- 对于小系统，mesolve 更高效。

### E.4.3 本文系统的选择

对于 2-atom Rydberg gate 系统：

- 每个原子有 3 个 relevant levels：$|g\rangle$，$|e\rangle$（intermediate），$|r\rangle$（Rydberg）。
- 2-atom Hilbert space dimension：$d = 3^2 = 9$（若包含 intermediate state）或 $d = 2^2 = 4$（adiabatic elimination 后）。
- mesolve 成本：$O(9^4) = O(6561)$ 或 $O(4^4) = O(256)$ per step——完全可行。

对于 3-atom 扩展（$d = 3^3 = 27$ 或 $d = 2^3 = 8$），mesolve 仍然可行。因此本文所有数值模拟均使用 mesolve。

## E.5 数值积分细节

本文使用 QuTiP (Quantum Toolbox in Python) 进行所有 Lindblad 方程的数值求解。关键实现细节如下：

**积分器选择**：QuTiP 的 mesolve 底层调用 SciPy 的 ODE integrators：

- **Adams method**（non-stiff solver）：适用于 Hamiltonian 变化平缓的情况。基于 predictor-corrector 方法，对于光滑问题具有高阶精度。
- **BDF method**（Backward Differentiation Formula，stiff solver）：适用于系统中存在多个差异很大的时间尺度时（如 fast Rabi oscillation + slow decay）。

**Time-dependent Hamiltonian**：当控制脉冲 $\Omega(t)$、$\Delta(t)$ 随时间变化时，使用 QuTiP 的 `QobjEvo` 对象，在离散时间点之间进行插值。插值方法影响精度：线性插值对于 piecewise-constant 脉冲已足够，cubic spline 插值适用于平滑脉冲。

**精度控制**：典型设置为 relative tolerance `rtol = 1e-8`，absolute tolerance `atol = 1e-10`。这些值在 gate fidelity 计算中提供优于 $10^{-6}$ 的数值精度，远超物理 decoherence 带来的 fidelity 限制。

## E.6 小结

Lindblad master equation (E.1) 及其 Kraus representation (E.2) 为 Rydberg gate 的 open-system simulation 提供了完整的理论框架。对于本文考虑的小规模系统（$d \leq 9$），mesolve 方法兼具高效性和精确性。数值参数的选择确保了计算结果的可靠性。

---

**参考文献**

- [QuTiP] J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP 2: A Python framework for the dynamics of open quantum systems," Comp. Phys. Comm. **184**, 1234 (2013).

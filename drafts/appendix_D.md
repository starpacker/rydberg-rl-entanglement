# 附录 D：GRAPE 梯度推导 + Barren Plateau

## D.1 引言

GRadient Ascent Pulse Engineering (GRAPE) 是一种广泛应用于量子最优控制的数值方法 [Khaneja05]。其核心思想是将控制脉冲离散化为分段常数函数，然后利用解析梯度公式高效优化。本附录完整推导 GRAPE 的梯度表达式，并讨论高维控制空间中的 barren plateau 问题。

## D.2 问题设定

考虑一个量子系统，其 Hamiltonian 在分段常数控制下为：

$$H(t) = H_0 + \sum_k u_k(t_j) H_k, \quad t \in [t_j, t_{j+1})$$

其中 $H_0$ 为 drift Hamiltonian，$H_k$ 为 control Hamiltonians，$u_k(t_j)$ 为第 $j$ 个时间步上第 $k$ 个控制参数的值。总时间 $T = N\Delta t$ 被等分为 $N$ 个时间步，每步时长 $\Delta t$。

在第 $j$ 个时间步内，Hamiltonian 为常数，因此 time evolution operator 为：

$$U_j = \exp\!\left[-i H(t_j) \Delta t\right] = \exp\!\left[-i\left(H_0 + \sum_k u_k(t_j) H_k\right)\Delta t\right]$$

总 propagator 为各时间步的有序乘积：

$$U = U_N U_{N-1} \cdots U_1$$

## D.3 Fidelity 定义

对于 state transfer 问题，目标是将初态 $|\psi_0\rangle$ 演化到目标态 $|\psi_{\text{tgt}}\rangle$。State fidelity 定义为：

$$\Phi = |\langle\psi_{\text{tgt}}|U|\psi_0\rangle|^2$$

$\Phi = 1$ 对应完美的 state transfer。GRAPE 的目标是找到控制参数 $\{u_k(t_j)\}$ 使得 $\Phi$ 最大化。

## D.4 梯度推导

**Step 1：引入辅助记号。** 定义 forward-propagated state 和 backward-propagated state：

$$|\phi_j\rangle = U_j U_{j-1} \cdots U_1 |\psi_0\rangle, \quad |\chi_j\rangle = U_{j+1}^\dagger U_{j+2}^\dagger \cdots U_N^\dagger |\psi_{\text{tgt}}\rangle$$

则 $\langle\psi_{\text{tgt}}|U|\psi_0\rangle = \langle\chi_j|U_j|\phi_{j-1}\rangle$（其中 $|\phi_0\rangle = |\psi_0\rangle$）。

**Step 2：对 $u_k(t_j)$ 求导。** 只有 $U_j$ 依赖于 $u_k(t_j)$，因此：

$$\frac{\partial}{\partial u_k(t_j)} \langle\psi_{\text{tgt}}|U|\psi_0\rangle = \langle\chi_j| \frac{\partial U_j}{\partial u_k(t_j)} |\phi_{j-1}\rangle$$

**Step 3：Matrix exponential 的导数。** 对于 $U_j = e^{-iH_j\Delta t}$，其中 $H_j = H_0 + \sum_k u_k(t_j)H_k$，求导得到：

$$\frac{\partial U_j}{\partial u_k(t_j)} = -i\Delta t\, H_k\, U_j + O(\Delta t^2)$$

严格地说，当 $[H_j, H_k] \neq 0$ 时，导数涉及 Duhamel formula：

$$\frac{\partial U_j}{\partial u_k(t_j)} = -i\Delta t \int_0^1 e^{-i(1-s)H_j\Delta t}\, H_k\, e^{-isH_j\Delta t}\, ds$$

但在 $\Delta t$ 足够小时（$\|H_j\|\Delta t \ll 1$），可以近似为 $\frac{\partial U_j}{\partial u_k(t_j)} \approx -i\Delta t\, H_k\, U_j$。这正是 GRAPE 中使用的标准近似。

**Step 4：组合得到 fidelity 的梯度。** 利用 $\Phi = |F|^2$ 其中 $F = \langle\psi_{\text{tgt}}|U|\psi_0\rangle$，由链式法则：

$$\frac{\partial\Phi}{\partial u_k(t_j)} = 2\,\text{Re}\!\left[\frac{\partial F}{\partial u_k(t_j)} \cdot F^*\right]$$

代入上述结果：

$$\frac{\partial\Phi}{\partial u_k(t_j)} = -i\Delta t \cdot 2\,\text{Re}\!\left[\langle\psi_{\text{tgt}}|U_N\cdots U_{j+1}\, H_k\, U_j \cdots U_1|\psi_0\rangle \cdot \langle\psi_0|U^\dagger|\psi_{\text{tgt}}\rangle\right] \tag{D.1}$$

等价地，用辅助记号写为：

$$\frac{\partial\Phi}{\partial u_k(t_j)} = -i\Delta t \cdot 2\,\text{Re}\!\left[\langle\chi_j|H_k|\phi_j\rangle \cdot F^*\right]$$

**计算效率**：关键在于 $|\phi_j\rangle$ 和 $|\chi_j\rangle$ 可以通过递推关系高效计算。一次 forward propagation 得到所有 $|\phi_j\rangle$，一次 backward propagation 得到所有 $|\chi_j\rangle$。因此，所有 $N \times K$ 个梯度分量（$N$ 个时间步、$K$ 个控制通道）只需 $O(N)$ 次 matrix-vector 乘法，外加 $N \times K$ 次内积运算。这比 finite-difference 方法（需要 $2NK$ 次 full propagation）高效得多。

## D.5 Barren Plateau 问题

GRAPE 的成功依赖于梯度提供有用的优化方向。然而，Larocca 等人 [Larocca22] 发现，在高维量子控制问题中存在严重的 barren plateau 现象。

**现象描述**：当 control landscape 的维度（$N \times K$）增大时，fidelity gradient $\partial\Phi/\partial u_k(t_j)$ 的方差指数级衰减：

$$\text{Var}\!\left[\frac{\partial\Phi}{\partial u_k}\right] \sim e^{-\alpha d}$$

其中 $d$ 为 Hilbert space dimension，$\alpha > 0$ 为常数。

**物理直觉**：在高维控制空间中，随机选取的初始控制参数会产生接近 Haar-random 的 unitary $U$。Haar-random unitary 将任意初态均匀分布在整个 Hilbert space 上，因此与特定 target state 的 overlap 为 $\sim 1/d$，其梯度也指数级小。

**实际后果**：

- 对于小系统（如 2-qubit gate，$d = 4$），barren plateau 不是问题，GRAPE 通常能收敛到 global optimum。
- 对于大系统（如 $d \geq 16$），GRAPE 可能陷入 local optima，远离 global optimum。
- 缓解策略包括：使用物理 informed 的初始猜测（而非随机初始化）、分层优化（先优化粗粒度再细化）、以及限制 control landscape 的有效维度 [Caneva11]。

## D.6 小结

GRAPE 梯度公式 (D.1) 提供了高效的 pulse optimization 方法，通过 forward-backward propagation 实现 $O(N)$ 复杂度的梯度计算。然而，在高维系统中，barren plateau 限制了 gradient-based 方法的有效性。对于本文考虑的 2-atom Rydberg gate（$d = 4$），GRAPE 仍然是有效的工具；但当扩展到更大系统时，可能需要结合 reinforcement learning 等 gradient-free 方法（参见附录 F 及主文第五章）。

---

**参考文献**

- [Khaneja05] N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent," J. Magn. Reson. **172**, 296 (2005).
- [Goerz14] M. H. Goerz, D. M. Reich, and C. P. Koch, "Optimal control theory for a unitary operation under dissipative evolution," New J. Phys. **16**, 055012 (2014).
- [Caneva11] T. Caneva, T. Calarco, and S. Montangero, "Chopped random-basis quantum optimization," Phys. Rev. A **84**, 022326 (2011).
- [Larocca22] M. Larocca et al., "Diagnosing barren plateaus with tools from quantum optimal control," Quantum **6**, 824 (2022).

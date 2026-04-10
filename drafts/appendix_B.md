# 附录 B：双光子绝热消除完整推导

## B.1 引言

在 Rydberg atom 实验中，从 ground state $|g\rangle$ 直接到 Rydberg state $|r\rangle$ 的 single-photon transition 通常在深紫外波段，实验上难以实现。实际方案通常采用 two-photon excitation，经过一个 intermediate excited state $|e\rangle$。当 intermediate detuning $\Delta$ 远大于其他能量尺度时，可以 adiabatically eliminate $|e\rangle$，将三能级系统约化为等效的二能级系统。本附录给出完整推导。

## B.2 三能级 Hamiltonian

考虑三能级系统 $\{|g\rangle, |e\rangle, |r\rangle\}$，其中 $|g\rangle$ 为 ground state（如 $^{87}\text{Rb}$ 的 $5S_{1/2}$），$|e\rangle$ 为 intermediate state（如 $5P_{3/2}$），$|r\rangle$ 为目标 Rydberg state（如 $nS$ 或 $nD$）。

在 rotating frame 中，经过 rotating-wave approximation (RWA)，系统 Hamiltonian 为（取 $\hbar = 1$）：

$$\hat{H} = \begin{pmatrix} 0 & \frac{\Omega_1}{2} & 0 \\ \frac{\Omega_1}{2} & -\Delta & \frac{\Omega_2}{2} \\ 0 & \frac{\Omega_2}{2} & -\delta \end{pmatrix} \tag{B.1}$$

各参数定义如下：

- $\Omega_1$：lower transition $|g\rangle \leftrightarrow |e\rangle$ 的 Rabi frequency（由第一束激光驱动，通常为 780 nm）。
- $\Omega_2$：upper transition $|e\rangle \leftrightarrow |r\rangle$ 的 Rabi frequency（由第二束激光驱动，通常为 480 nm）。
- $\Delta$：intermediate detuning，即第一束激光频率与 $|g\rangle \to |e\rangle$ transition frequency 之差。$\Delta > 0$ 表示 blue detuning。
- $\delta$：two-photon detuning，即两束激光频率之和与 $|g\rangle \to |r\rangle$ transition frequency 之差。

注意 Hamiltonian 中 $|g\rangle$ 和 $|r\rangle$ 之间没有直接耦合项（因为 electric dipole selection rules 禁止 $\Delta\ell = 0$ 或 $\Delta\ell = 2$ 的 single-photon transition）。

## B.3 Schrödinger Equation

将系统 state 展开为 $|\psi(t)\rangle = c_g(t)|g\rangle + c_e(t)|e\rangle + c_r(t)|r\rangle$，代入 time-dependent Schrödinger equation $i\dot{|\psi\rangle} = \hat{H}|\psi\rangle$（已取 $\hbar = 1$），得到三个耦合方程：

$$i\dot{c}_g = \frac{\Omega_1}{2} c_e \tag{B.3a}$$

$$i\dot{c}_e = \frac{\Omega_1}{2} c_g - \Delta\, c_e + \frac{\Omega_2}{2} c_r \tag{B.3b}$$

$$i\dot{c}_r = \frac{\Omega_2}{2} c_e - \delta\, c_r \tag{B.3c}$$

## B.4 Adiabatic Elimination

当 intermediate detuning 满足 $|\Delta| \gg |\Omega_1|, |\Omega_2|, \gamma_e$（其中 $\gamma_e$ 为 $|e\rangle$ 的 spontaneous emission rate），intermediate state 的 population 始终很小（$|c_e|^2 \ll 1$），其振幅 adiabatically 跟随 $c_g$ 和 $c_r$ 的变化。

**Step 1**：设 $\dot{c}_e \approx 0$（adiabatic condition），从式 (B.3b) 得：

$$0 \approx \frac{\Omega_1}{2} c_g - \Delta\, c_e + \frac{\Omega_2}{2} c_r$$

解出 $c_e$：

$$c_e = \frac{\Omega_1 c_g + \Omega_2 c_r}{2\Delta} \tag{B.4a}$$

**Step 2**：将式 (B.4a) 代入式 (B.3a)：

$$i\dot{c}_g = \frac{\Omega_1}{2} \cdot \frac{\Omega_1 c_g + \Omega_2 c_r}{2\Delta} = \frac{\Omega_1^2}{4\Delta} c_g + \frac{\Omega_1 \Omega_2}{4\Delta} c_r$$

**Step 3**：将式 (B.4a) 代入式 (B.3c)：

$$i\dot{c}_r = \frac{\Omega_2}{2} \cdot \frac{\Omega_1 c_g + \Omega_2 c_r}{2\Delta} - \delta\, c_r = \frac{\Omega_1 \Omega_2}{4\Delta} c_g + \left(\frac{\Omega_2^2}{4\Delta} - \delta\right) c_r$$

## B.5 Effective Two-Level Hamiltonian

将上述两个方程写成矩阵形式 $i\dot{\mathbf{c}} = \hat{H}_{\text{eff}} \mathbf{c}$，其中 $\mathbf{c} = (c_g, c_r)^T$：

$$\hat{H}_{\text{eff}} = \begin{pmatrix} \frac{\Omega_1^2}{4\Delta} & \frac{\Omega_1\Omega_2}{4\Delta} \\ \frac{\Omega_1\Omega_2}{4\Delta} & -\delta + \frac{\Omega_2^2}{4\Delta} \end{pmatrix} \tag{B.2}$$

从此 effective Hamiltonian 中提取两个关键物理量：

### Effective Rabi Frequency

off-diagonal 元素给出 effective two-photon Rabi frequency：

$$\Omega_{\text{eff}} = \frac{\Omega_1\Omega_2}{2\Delta} \tag{B.3}$$

这是一个 second-order process：通过 virtual excitation of $|e\rangle$ 实现 $|g\rangle \leftrightarrow |r\rangle$ 的等效耦合。$\Omega_{\text{eff}}$ 正比于两个单光子 Rabi frequencies 的乘积，反比于 detuning $\Delta$。

### Differential AC Stark Shift

diagonal 元素中的 light shift 项不相等，产生 differential AC Stark shift：

$$\delta_{\text{AC}} = \frac{\Omega_1^2 - \Omega_2^2}{4\Delta} \tag{B.4}$$

这一 shift 修正了 effective two-photon resonance condition。要实现精确的 two-photon resonance（$|g\rangle \leftrightarrow |r\rangle$），需要将 $\delta$ 设为 $\delta_{\text{AC}}$ 而非零。在实验中，通常通过调节激光频率来补偿此 shift。

## B.6 Schrieffer-Wolff 变换视角

Adiabatic elimination 可以从更系统的 Schrieffer-Wolff transformation 角度理解。定义 projection operators：

- $P$：投影到低能子空间 $\{|g\rangle, |r\rangle\}$
- $Q = 1 - P$：投影到高能子空间 $\{|e\rangle\}$

将 Hamiltonian 分块：$H = H_0 + V$，其中 $H_0$ 包含对角项，$V$ 包含耦合项。Effective Hamiltonian 在二阶微扰下为：

$$H_{\text{eff}} = PHP + PHQ\frac{1}{E - QHQ}QHP$$

对于我们的三能级系统，$QHQ = -\Delta$（取 $|e\rangle$ 的对角能量），$PHQ$ 和 $QHP$ 分别包含 $\Omega_1/2$ 和 $\Omega_2/2$ 的耦合矩阵元素。代入后恢复式 (B.2) 的结果。

Schrieffer-Wolff 变换的优势在于它可以系统地推广到更高阶修正，以及处理更复杂的多能级系统（如考虑 hyperfine structure 时多个 intermediate states 的情况）。

## B.7 Intermediate State 散射率

虽然 $|e\rangle$ 被 adiabatically eliminated，但它并非完全"不参与"：finite population $|c_e|^2 > 0$ 与 spontaneous emission rate $\gamma_e$ 结合，产生不可忽略的 photon scattering rate。

从式 (B.4a)，在 two-photon resonance 附近且 $|c_r| \ll |c_g|$ 时，$|c_e|^2 \approx \Omega_1^2/(4\Delta^2)$。因此散射率为：

$$R_{\text{sc}} = \gamma_e |c_e|^2 \approx \frac{\gamma_e \Omega_1^2}{4\Delta^2} \tag{B.5}$$

此式给出了 adiabatic elimination 的核心 trade-off：

- **增大 $\Delta$** 可减小 $R_{\text{sc}}$（减少 decoherence），但同时减小 $\Omega_{\text{eff}} \propto 1/\Delta$（减慢 gate 速度）。
- **增大 $\Omega_1, \Omega_2$** 可增大 $\Omega_{\text{eff}}$，但 $R_{\text{sc}} \propto \Omega_1^2$。

实际优化中，常用 figure of merit $\Omega_{\text{eff}}/R_{\text{sc}} = \Omega_2/(2\gamma_e)$，表明提高 upper transition 的 Rabi frequency $\Omega_2$ 是最有效的策略 [dL18]。

## B.8 小结

通过 adiabatic elimination，三能级 two-photon excitation 问题被简化为一个等效的 two-level system，其中 effective Rabi frequency $\Omega_{\text{eff}} = \Omega_1\Omega_2/(2\Delta)$，代价是引入 AC Stark shift $\delta_{\text{AC}}$ 和 photon scattering $R_{\text{sc}}$。这一结果是后续分析 Rydberg blockade gate dynamics（参见主文第三章）和优化控制脉冲（参见附录 D）的基础。

---

**参考文献**

- [dL18] A. de Léséleuc et al., "Analysis of imperfections in the coherent optical excitation of single atoms to Rydberg states," Phys. Rev. A **97**, 053803 (2018).
- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," Rev. Mod. Phys. **82**, 2313 (2010).

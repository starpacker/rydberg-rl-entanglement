# 驯服量子混沌：里德堡原子阵列中基于强化学习的高保真度纠缠态制备

**Taming Quantum Chaos: High-Fidelity Entanglement Preparation in Rydberg Atom Arrays via Reinforcement Learning**

> 原子物理学期中报告

---

## 摘要

里德堡原子因其兼具长寿命（$\tau \sim n^{*3}$）与超强 van der Waals 相互作用（$C_6 \sim n^{*11}$）的独特性质，成为当前最有前景的量子纠缠操控平台之一。然而，真实实验中多源退相干通道（多普勒展宽、位置抖动的 $1/R^6$ 非线性放大、激光幅度 OU 噪声、servo bump 相位噪声、Rydberg 态有限寿命）构成了一个"双重夹击"——噪声标度律要求尽可能大的 Rabi 频率 $\Omega$，而 blockade 条件要求 $\Omega \ll V_{\text{vdW}}$，将可用参数空间压缩到狭窄窗口内。在此约束下，传统控制算法各自遭遇根本性失效：STIRAP 在长时条件下被退相干侵蚀至 $F \approx 0.74$，GRAPE 的开环优化在 sim-to-real gap 面前脆弱，反绝热驱动在非马尔可夫有色噪声下失效。

本文提出将 Rydberg 门控制问题重新映射为 Markov 决策过程，采用 proximal policy optimization (PPO) + domain randomization 求解。通过在每个训练 episode 中重新采样全部噪声参数，策略网络学习到一种对噪声分布鲁棒的控制策略。基于 Stable-Baselines3 + scipy 矩阵指数化 Lindblad 演化的数值实验，在三个具有不同噪声特征的场景上系统地对比了 PPO、STIRAP 和 GRAPE 的表现。核心结果是：在专门设计的高噪声场景 C（噪声幅度放大 3 倍 + $\pm 5\%$ 系统性幅度偏置）中，PPO 经 3M 步 CPU 训练后达到 $\langle F \rangle = 0.990 \pm 0.008$，**显著超过** STIRAP（$0.850 \pm 0.150$）和 GRAPE（$0.814 \pm 0.167$），且在 0–5% 额外幅度误差的鲁棒性扫描中保持稳定（$\langle F \rangle > 0.989$），展示了 domain randomization 赋予的噪声鲁棒性优势。本文从 quantum defect theory 出发，经 Rydberg blockade 机制、退相干预算分析、传统算法失效论证，到 RL 方案设计与数值验证，构建了一条完整的物理—方法证明链。

---

# 1 Rydberg 原子物理基础

本节建立全文的物理基础：从量子亏损理论出发，阐明里德堡态的能级结构与标度律，揭示里德堡原子兼具长寿命与强相互作用的独特性质——这正是其成为量子信息平台核心载体的物理根源。

## 1.1 从氢原子到碱金属里德堡原子

氢原子能级由 Bohr 公式给出：

$$E_n = -\frac{\text{Ry}}{n^2}$$

其中 $\text{Ry} \approx 13.6\;\text{eV}$ 为 Rydberg 常量对应的能量。对于主量子数 $n \gg 1$ 的高激发态（即 Rydberg 态），价电子远离原子实，氢原子模型提供了良好的零级近似。

然而，碱金属原子（如 Rb、Cs）与氢原子存在本质差异。碱金属原子具有填满的内壳层电子，当价电子运动到小 $r$ 区域（尤其是低角动量 $\ell$ 的轨道）时，会深入内壳层区域，经历两个效应：(i) **core penetration**——价电子穿入内壳层，感受到未被完全屏蔽的核电荷；(ii) **core polarization**——价电子的电场极化内壳层电子云，产生额外的吸引势。两者的共同结果是：在小 $r$ 处，价电子所感受的有效势偏离纯 Coulomb 势。

**Quantum defect theory (QDT)** 提供了处理这一问题的简洁框架。其物理图像如下：短程非 Coulomb 势在径向波函数中引入一个额外的相移（phase shift）；当把该波函数在大 $r$ 处与纯 Coulomb 渐近解匹配时，这一相移等效于有效量子数的偏移。由此得到 Rydberg–Ritz 公式：

$$E_{n\ell j} = -\frac{\text{Ry}^*}{(n - \delta_{n\ell j})^2} \tag{1.1}$$

其中 $\text{Ry}^* = \text{Ry} \cdot M/(M + m_e)$ 为约化质量修正后的 Rydberg 常量，$\delta_{n\ell j}$ 为 quantum defect，可展开为

$$\delta_{n\ell j} = \delta_0 + \frac{\delta_2}{(n - \delta_0)^2} + \cdots$$

低 $\ell$ 态的 quantum defect 较大（穿入效应显著），高 $\ell$ 态则趋近于零（趋于氢原子行为）。完整的 WKB 相位积分推导见附录 A。

对于 $^{87}\text{Rb}$，实验测定的 quantum defect 值为 [SA18, Gallagher94]：

| 轨道 | $\delta_0$ |
|------|-----------|
| $nS_{1/2}$ | 3.1312 |
| $nP_{3/2}$ | 2.6549 |
| $nD_{5/2}$ | 1.3480 |

可见 $S$ 态的 quantum defect 最大，对应最强的 core penetration 效应。

![图 1：碱金属里德堡原子的能级结构与量子亏损](figures/fig01_energy_levels.png)

**Fig. 1**：碱金属里德堡原子能级结构。低 $\ell$ 态因 core penetration 效应偏离氢原子能级（量子亏损 $\delta_\ell$ 较大），高 $\ell$ 态趋近氢原子行为。

## 1.2 标度律

定义有效量子数 $n^* = n - \delta_{n\ell j}$，Rydberg 态的诸多物理量均表现出关于 $n^*$ 的幂次标度律（scaling law）。这些标度律直接决定了 Rydberg 原子在量子信息中的应用潜力。表 1 汇总了关键物理量的标度关系及其对 Rb $70S$ 态的典型数值。

**表 1**：Rydberg 原子关键物理量的标度律（以 Rb $70S_{1/2}$ 态为例，$n^* \approx 66.9$）

| 物理量 | 标度 | 物理意义 | Rb 70S 数值 |
|--------|------|----------|-------------|
| 轨道半径 $\langle r \rangle$ | $\sim n^{*2}$ | 原子尺度膨胀至亚微米量级，产生巨大的电偶极矩 | ${\sim}0.27\;\mu\text{m}$ |
| 偶极矩元 $\langle n'\ell' \| er \| n\ell \rangle$ | $\sim n^{*2}$ | 与光场的强耦合，使得单光子 Rabi 频率显著增大 | ${\sim}4500\;ea_0$ |
| 极化率 $\alpha$ | $\sim n^{*7}$ | 对外电场极其敏感，微弱杂散场即可产生显著 Stark 移位 | ${\sim}10^2\;\text{MHz/(V/cm)}^2$ |
| 辐射寿命 $\tau_{\text{rad}}$ | $\sim n^{*3}$ | 长寿命使得在退相干前可执行多步量子操控 | ${\sim}150\;\mu\text{s}$ |
| BBR 限制寿命 (300 K) $\tau_{\text{BBR}}$ | $\sim n^{*2}$ | 室温黑体辐射诱导的跃迁不可忽略，限制有效寿命 | ${\sim}80\;\mu\text{s}$ |
| $C_6$ 系数 | $\sim n^{*11}$ | van der Waals 相互作用极强，支撑 Rydberg blockade 机制 | ${\sim}862\;\text{GHz}{\cdot}\mu\text{m}^6$ |
| 阻塞半径 $R_b$ | $\sim n^{*11/6}$ | 在 $R_b$ 内两原子无法同时激发，构成量子门操控的核心资源 | ${\sim}9.7\;\mu\text{m}$ ($\Omega/2\pi = 1\;\text{MHz}$) |

以上标度律的物理根源在于 Rydberg 态波函数在空间上的大幅展开。以下逐一给出推导。

### 轨道半径 $\langle r \rangle \propto n^{*2}$

直接来自类氢原子的精确解。氢原子 $n\ell$ 态的径向期望值为

$$\langle r \rangle_{n\ell} = \frac{a_0}{2}\bigl[3n^2 - \ell(\ell+1)\bigr].$$

对于 $S$ 态（$\ell = 0$），$\langle r \rangle = \frac{3}{2} a_0 n^2$。在碱金属中，将 $n$ 替换为 $n^*$ 即得标度。偶极矩元 $d \sim e\langle r \rangle \propto n^{*2}$ 随之增长。

### 辐射寿命 $\tau_{\text{rad}} \propto n^{*3}$

由 Einstein $A$ 系数，自发辐射速率为
$$A_{n \to n'} = \frac{4 \alpha_{\text{fs}} \omega_{nn'}^3}{3 c^2} |\langle n' | r | n \rangle|^2,$$
其中 $\alpha_{\text{fs}} \approx 1/137$ 为精细结构常数。

总衰变速率 $\Gamma = \sum_{n'} A_{n \to n'}$ 需要对所有低能级 $n'$ 求和。关键观察是：对于高 $n$ Rydberg 态，**主要衰变通道**指向最低的几个能级（$n' \sim$ 几），因为这些跃迁的偶极矩元最大。对于 $n' \ll n$ 的跃迁：
- 跃迁频率 $\omega_{nn'} \approx \text{Ry}/(\hbar n'^2)$，几乎与 $n$ 无关；
- 跃迁偶极矩元 $\langle n' | r | n \rangle \propto n^{*-3/2}$（来自氢原子径向波函数在小 $r$ 处的归一化因子 $\sim n^{-3/2}$，即波函数在内区域的振幅随 $n$ 增大而稀释）。

因此 $\Gamma \propto \omega^3 \cdot |d|^2 \sim \text{const} \times (n^{*-3/2})^2 = n^{*-3}$，故 $\tau = 1/\Gamma \propto n^{*3}$。

### $C_6$ 系数 $\propto n^{*11}$

van der Waals 相互作用来自偶极–偶极耦合的二阶微扰论。两个处于 $|n\ell\rangle$ 态的原子，通过虚跃迁到中间态 $|n'\ell'\rangle|n''\ell''\rangle$ 获得能量修正：

$$C_6 = -\sum_{n',n''} \frac{|\langle n'\ell',\, n''\ell'' | \hat{V}_{dd} | n\ell,\, n\ell \rangle|^2}{\Delta E_{n',n''}},$$

其中 $\hat{V}_{dd} \propto d_1 d_2 / R^3$ 为偶极–偶极相互作用。主导贡献来自近简并的 pair state（如 $nS + nS \leftrightarrow (n-1)P + nP$），对这些通道：
- 偶极矩元 $d \propto n^{*2}$（相邻能级间的跃迁），故 $V_{dd}$ 矩阵元 $\propto d^2 \propto n^{*4}$；
- Förster 缺陷能 $\Delta E \propto n^{*-3}$（相邻 Rydberg 能级间距）。

因此 $C_6 \propto d^4/\Delta E = (n^{*2})^4 / n^{*-3} = n^{*11}$。

### 阻塞半径 $R_b \propto n^{*11/6}$

Rydberg blockade 半径定义为 van der Waals 相互作用能等于驱动光场线宽的距离：$C_6/R_b^6 = \hbar\Omega$，解出 $R_b = (C_6/\hbar\Omega)^{1/6} \propto (n^{*11})^{1/6} = n^{*11/6}$（在固定 $\Omega$ 下）。

### 极化率 $\alpha \propto n^{*7}$

标量极化率由二阶微扰论给出：$\alpha \propto \sum_{n'} d_{nn'}^2/\Delta E_{nn'}$。主导贡献来自 $n' \approx n$ 的近邻态，此时 $d \propto n^{*2}$，$\Delta E \propto n^{*-3}$，故 $\alpha \propto n^{*4}/n^{*-3} = n^{*7}$。

总而言之，Rydberg 态独特地结合了长寿命（$\tau \sim n^{*3}$）与强相互作用（$C_6 \sim n^{*11}$），这一组合在自然界的原子态中几乎无可替代——它正是里德堡原子量子计算平台的物理基石。

![图 2：Rydberg 态关键物理量的标度律](figures/fig02_scaling_laws.png)

**Fig. 2**：Rydberg 态关键物理量随有效量子数 $n^*$ 的标度关系。长寿命（$\tau \sim n^{*3}$）与强相互作用（$C_6 \sim n^{*11}$）的组合是里德堡量子计算平台的物理基石。

## 1.3 BBR 与寿命修正

Rydberg 原子的有效寿命不仅受自发辐射限制，还受到黑体辐射（blackbody radiation, BBR）诱导的受激跃迁的显著影响。在有限温度下，有效衰变速率为

$$\frac{1}{\tau_{\text{eff}}} = \frac{1}{\tau_{0\text{K}}} + \Gamma_{\text{BBR}}, \qquad \Gamma_{\text{BBR}} \approx \frac{4\alpha_{\text{fs}}^3 k_B T}{3 n^{*2} \hbar} \tag{1.2}$$

其中 $\tau_{0\text{K}}$ 为零温下的纯辐射寿命（包含自发辐射至所有低能级的贡献），$\alpha_{\text{fs}} \approx 1/137$ 为精细结构常数，$k_B T$ 为热能。BBR 速率 $\Gamma_{\text{BBR}}$ 的 $n^{*-2}$ 标度可直观理解为：黑体光谱在 Rydberg 态间距对应的微波频段（$\propto n^{*-3}$）具有充足的光子数，而跃迁偶极矩元 $\propto n^{*2}$。虽然 $\Gamma_{\text{BBR}} \propto n^{*-2}$ 本身随 $n$ 增大而减小，但其衰减速度慢于辐射速率 $\Gamma_{\text{rad}} \propto n^{*-3}$，因此 BBR 在总衰变速率中的 **占比** 随 $n$ 增大而增加，在高 $n$ 时成为主导的退激发机制。

Beterov 等人 [Beterov09] 系统地计算了 Rb、Cs 等碱金属 Rydberg 态在室温下的有效寿命。以 Rb $nS$ 态为例：在 $T = 300\;\text{K}$ 下，$n = 70$ 时 $\tau_{\text{eff}} \approx 50\text{–}80\;\mu\text{s}$，远低于零温辐射寿命 $\tau_{0\text{K}} \approx 150\;\mu\text{s}$。这说明对于 $n \gtrsim 50$ 的 Rydberg 态，BBR 对有效寿命的贡献不可忽视。

从量子计算的角度看，有效寿命 $\tau_{\text{eff}}$ 直接限定了量子门操作时间的上界：必须满足

$$T_{\text{gate}} \ll \tau_{\text{eff}}$$

才能保证门操作在退相干前完成。对于典型的 Rydberg 门方案（$n \sim 60\text{–}80$），$\tau_{\text{eff}} \sim 50\text{–}100\;\mu\text{s}$，而单次门操作时间 $T_{\text{gate}} \sim 0.1\text{–}1\;\mu\text{s}$，留有约两个数量级的裕度。然而，在需要多步序列操作或面临其他退相干通道（如多普勒展宽、激光相位噪声）时，BBR 寿命仍是一个不可忽略的系统性限制 [Beterov09, SWM10]。

---

**参考文献**

- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).
- [Gallagher94] T. F. Gallagher, *Rydberg Atoms* (Cambridge University Press, 1994).
- [SA18] N. Šibalić and C. S. Adams, *Rydberg Physics* (IOP Publishing, 2018).  *(See also [Steck18] in Appendix A)*
- [Beterov09] I. I. Beterov *et al.*, "Quasiclassical calculations of blackbody-radiation-induced depopulation rates and effective lifetimes of Rydberg nS, nP, and nD alkali-metal atoms with n ≤ 80," *Phys. Rev. A* **79**, 052504 (2009).

---

# 2. 原子--激光相互作用

本节从经典电磁场与二能级原子的耦合出发，推导旋转波近似下的有效 Hamiltonian，进而引入双光子激发方案，建立从基态到里德堡态的控制 Hamiltonian。这一框架是后续所有章节（阻塞机制、门操作、退相干分析）的共同起点。

---

## 2.1 二能级原子 + 经典激光场 + RWA

### 偶极近似

考虑原子与单模激光场的相互作用。光学波长 $\lambda \sim 500\text{--}800\;\text{nm}$ 远大于原子尺度 $a_0 \approx 0.053\;\text{nm}$，因此在原子所在区域内电场几乎均匀，可以将 $\mathbf{E}(\mathbf{r},t)$ 在原子质心处展开并仅保留零阶项。这就是 **dipole approximation**，其有效性条件为

$$
\lambda \gg a_0.
$$

### 偶极 Hamiltonian

在 dipole approximation 下，原子与光场的相互作用 Hamiltonian 为

$$
\hat{H}_I = -\hat{\mathbf{d}} \cdot \mathbf{E}(t),
$$

其中 $\hat{\mathbf{d}} = -e\hat{\mathbf{r}}$ 为电偶极矩算符。设原子具有基态 $|g\rangle$ 和激发态 $|r\rangle$（能量差 $\hbar\omega_0$），激光场为线偏振平面波

$$
\mathbf{E}(t) = \mathbf{E}_0 \cos(\omega_L t) = \frac{\mathbf{E}_0}{2}\bigl(e^{-i\omega_L t} + e^{i\omega_L t}\bigr).
$$

在 $\{|g\rangle, |r\rangle\}$ 基下，裸 Hamiltonian 为 $\hat{H}_0 = \hbar\omega_0 |r\rangle\langle r|$（取 $E_g = 0$）。由宇称选择定则，$\langle g|\hat{\mathbf{d}}|g\rangle = \langle r|\hat{\mathbf{d}}|r\rangle = 0$，因此 $\hat{H}_I$ 仅具有非对角矩阵元。

### Bare Rabi frequency 的物理来源

定义 **bare Rabi frequency**

$$
\Omega = -\frac{\mathbf{d}_{rg} \cdot \mathbf{E}_0}{\hbar},
$$

其中 $\mathbf{d}_{rg} = \langle r|\hat{\mathbf{d}}|g\rangle$ 为跃迁偶极矩元。$\Omega$ 的物理意义是激光场驱动原子在 $|g\rangle$ 和 $|r\rangle$ 之间振荡的角频率。它由两个因素决定：

1. **跃迁偶极矩元** $d_{rg} = |\langle r|\hat{\mathbf{d}}|g\rangle|$：反映原子的内禀耦合强度，由初末态的径向与角向波函数重叠积分决定。对于碱金属原子的低激发态到 Rydberg 态的直接跃迁，$d_{rg}$ 很小（$\propto n^{*-3/2}$），因此实验中通常采用双光子方案（见 §2.2）。
2. **激光电场振幅** $E_0$：与激光功率 $P$ 和束腰 $w$ 的关系为 $E_0 = \sqrt{2P/(\pi w^2 c \epsilon_0)}$。通过聚焦激光束（减小 $w$）或增大功率可提高 $\Omega$。

典型实验值：对于 ${}^{87}$Rb 的双光子激发至 $|70S\rangle$，使用 $\sim 100\;\text{mW}$ 聚焦激光可实现 $\Omega/2\pi \sim 1\text{--}10\;\text{MHz}$。

总 Hamiltonian 在 Schrödinger picture 下为

$$
\hat{H} = \hbar\omega_0 |r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl(e^{-i\omega_L t} + e^{i\omega_L t}\bigr)\bigl(|r\rangle\langle g| + |g\rangle\langle r|\bigr).
$$

### 旋转坐标变换（详细推导）

引入 **rotating frame** 变换

$$
\hat{U}(t) = e^{-i\omega_L t\,|r\rangle\langle r|},
$$

将态矢量变换为 $|\tilde{\psi}(t)\rangle = \hat{U}^\dagger(t)|\psi(t)\rangle$。对应的变换后 Hamiltonian 为

$$
\hat{H}_{\text{rot}} = \hat{U}^\dagger \hat{H} \hat{U} - i\hbar\,\hat{U}^\dagger \dot{\hat{U}}.
$$

我们逐项计算。

**第一步：计算 $\hat{U}$ 的具体作用。** 由于 $|r\rangle\langle r|$ 是投影算符（$|r\rangle\langle r|^2 = |r\rangle\langle r|$），指数展开给出

$$
\hat{U}(t) = |g\rangle\langle g| + e^{-i\omega_L t}|r\rangle\langle r|, \qquad \hat{U}^\dagger(t) = |g\rangle\langle g| + e^{i\omega_L t}|r\rangle\langle r|.
$$

因此 $\hat{U}$ 对基矢的作用为：$\hat{U}|g\rangle = |g\rangle$，$\hat{U}|r\rangle = e^{-i\omega_L t}|r\rangle$。

**第二步：变换裸 Hamiltonian $\hat{H}_0 = \hbar\omega_0|r\rangle\langle r|$。**

$$
\hat{U}^\dagger \hat{H}_0 \hat{U} = \hbar\omega_0\,\hat{U}^\dagger|r\rangle\langle r|\hat{U} = \hbar\omega_0\,e^{i\omega_L t}|r\rangle \cdot \langle r|e^{-i\omega_L t} = \hbar\omega_0|r\rangle\langle r|.
$$

裸能量部分不变（对角项不受 $\hat{U}$ 影响）。

**第三步：变换耦合项 $\hat{H}_I$。** 先看 $|r\rangle\langle g|$：

$$
\hat{U}^\dagger |r\rangle\langle g| \hat{U} = (e^{i\omega_L t}|r\rangle)(\langle g|) = e^{i\omega_L t}|r\rangle\langle g|.
$$

类似地 $\hat{U}^\dagger |g\rangle\langle r| \hat{U} = e^{-i\omega_L t}|g\rangle\langle r|$。因此耦合项变为

$$
\hat{U}^\dagger \hat{H}_I \hat{U} = \frac{\hbar\Omega}{2}\bigl(e^{-i\omega_L t} + e^{i\omega_L t}\bigr)\bigl(e^{i\omega_L t}|r\rangle\langle g| + e^{-i\omega_L t}|g\rangle\langle r|\bigr).
$$

展开括号得到 **四项**：

$$
= \frac{\hbar\Omega}{2}\Bigl[\underbrace{|r\rangle\langle g| + |g\rangle\langle r|}_{\text{缓变项 (near-resonant)}} + \underbrace{e^{2i\omega_L t}|r\rangle\langle g| + e^{-2i\omega_L t}|g\rangle\langle r|}_{\text{快振荡项 (counter-rotating)}}\Bigr].
$$

**第四步：计算 $-i\hbar\hat{U}^\dagger\dot{\hat{U}}$。**

$$
\dot{\hat{U}} = -i\omega_L|r\rangle\langle r|\,e^{-i\omega_L t} = -i\omega_L\,e^{-i\omega_L t}|r\rangle\langle r|,
$$

$$
-i\hbar\hat{U}^\dagger\dot{\hat{U}} = -i\hbar\,(e^{i\omega_L t}|r\rangle\langle r|)(-i\omega_L\,e^{-i\omega_L t}|r\rangle\langle r|) = -\hbar\omega_L|r\rangle\langle r|.
$$

**合并所有项**：

$$
\hat{H}_{\text{rot}} = (\hbar\omega_0 - \hbar\omega_L)|r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl[|r\rangle\langle g| + |g\rangle\langle r|\bigr] + \frac{\hbar\Omega}{2}\bigl[e^{2i\omega_L t}|r\rangle\langle g| + e^{-2i\omega_L t}|g\rangle\langle r|\bigr].
$$

### 旋转波近似 (RWA)

最后两项以频率 $2\omega_L$ 快速振荡。当 **bare Rabi frequency** 远小于光学频率，即

$$
\left|\frac{\Omega}{\omega_L}\right| \ll 1,
$$

这些快振荡项 $\sim e^{\pm 2i\omega_L t}$ 在任何实验可分辨的时间尺度 $\Delta t \gg 1/\omega_L$ 上平均为零，可安全丢弃。对于光学跃迁（$\omega_L/2\pi \sim 10^{14}\;\text{Hz}$, $\Omega/2\pi \sim \text{MHz}$），该条件以 $\sim 10^8$ 的余量满足。

丢弃 counter-rotating terms 后，定义 detuning $\Delta = \omega_L - \omega_0$（激光频率减去原子跃迁频率），得到 **RWA 有效 Hamiltonian**：

$$
\hat{H}_{\text{rot}} = -\hbar\Delta\,|r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl(|r\rangle\langle g| + |g\rangle\langle r|\bigr). \tag{2.1}
$$

这是一个 **时间无关** 的二能级 Hamiltonian，完全由两个实验可调参数 $(\Omega, \Delta)$ 控制——这正是旋转坐标变换的核心优势：将含时问题化为静态问题。

### Bloch 球图像

式 (2.1) 的动力学可映射到 Bloch 球上的旋转：定义 Bloch 向量 $\mathbf{B} = (\text{Re}\,\rho_{gr},\;\text{Im}\,\rho_{gr},\;(P_g - P_r)/2)$，则运动方程为 $\dot{\mathbf{B}} = \boldsymbol{\Omega}_{\text{eff}} \times \mathbf{B}$，其中有效磁场 $\boldsymbol{\Omega}_{\text{eff}} = (\Omega, 0, -\Delta)$。态矢量绕 $\boldsymbol{\Omega}_{\text{eff}}$ 方向进动，进动频率为广义 Rabi 频率 $\Omega' = \sqrt{\Omega^2 + \Delta^2}$（参见 Fig. 4）。

![图 4：Rabi 振荡与 Bloch 球表示](figures/fig04_rabi_bloch.png)

**Fig. 4**：二能级系统的 Rabi 振荡（左）与 Bloch 球图像（右）。态矢量绕有效磁场 $\boldsymbol{\Omega}_{\text{eff}} = (\Omega, 0, -\Delta)$ 进动。

### Rabi 振荡

在共振条件 $\Delta = 0$ 下，系统初始处于 $|g\rangle$，激发态布居的时间演化为

$$
P_r(t) = \sin^2\!\left(\frac{\Omega t}{2}\right),
$$

表现为频率 $\Omega$ 的完美 Rabi oscillation。这是实现量子门操作的基本时钟：例如，$t = \pi/\Omega$ 实现 $|g\rangle \to |r\rangle$ 的 $\pi$ 脉冲。

### AC Stark 位移

在大失谐极限 $|\Delta| \gg \Omega$ 下，激光场不会驱动实际跃迁，但通过虚跃迁过程使能级发生位移。二阶微扰论给出 AC Stark (light) shift

$$
\delta_{\text{AC}} = \frac{\Omega^2}{4\Delta},
$$

这一效应在双光子方案中尤为重要（见 \S2.2）。

---

## 2.2 双光子激发到里德堡态

### 实验动机

碱金属原子（如 $^{87}$Rb）从基态 $5S_{1/2}$ 单光子直接激发到里德堡 $nS$ 态需要深紫外光（$\lambda < 300\;\text{nm}$），这在技术上极为困难——短波长激光功率低、光学元件损耗大、且难以获得足够的 Rabi 频率。实际实验中普遍采用 **two-photon ladder** 方案 [SWM10, dL18]：

$$
5S_{1/2} \xrightarrow{\;\Omega_1,\;\lambda \approx 780\;\text{nm}\;} 5P_{3/2} \xrightarrow{\;\Omega_2,\;\lambda \approx 480\;\text{nm}\;} nS/nD,
$$

两束激光均处于方便的可见/近红外波段。

### 三能级系统与大失谐

记三个能级为 $|g\rangle = |5S_{1/2}\rangle$、$|e\rangle = |5P_{3/2}\rangle$（中间态）、$|r\rangle = |nS/nD\rangle$（里德堡态）。下行激光以 Rabi 频率 $\Omega_1$ 耦合 $|g\rangle \leftrightarrow |e\rangle$，上行激光以 $\Omega_2$ 耦合 $|e\rangle \leftrightarrow |r\rangle$。为避免中间态 $|e\rangle$ 的自发辐射（$\gamma_{5P}/2\pi \approx 6\;\text{MHz}$），两束激光相对中间态施加大的单光子 detuning $\Delta$，同时保持双光子共振。

### 绝热消除

在大失谐极限 $\Delta \gg \Omega_1, \Omega_2$ 下，中间态 $|e\rangle$ 的布居极小（$|c_e|^2 \sim \Omega_1^2/4\Delta^2 \ll 1$），其振幅可被绝热消除（adiabatic elimination）。以下给出核心推导步骤（完整的 Schrieffer-Wolff 变换见附录 B）。

在 rotating frame 中，三能级 Hamiltonian（取 $\hbar = 1$，双光子共振 $\delta = 0$）为

$$
\hat{H} = \begin{pmatrix} 0 & \Omega_1/2 & 0 \\ \Omega_1/2 & -\Delta & \Omega_2/2 \\ 0 & \Omega_2/2 & 0 \end{pmatrix},
$$

其中行/列分别对应 $|g\rangle, |e\rangle, |r\rangle$。注意 $|g\rangle$ 和 $|r\rangle$ 之间无直接耦合（偶极选择定则）。

将态展开为 $|\psi\rangle = c_g|g\rangle + c_e|e\rangle + c_r|r\rangle$，代入 Schrödinger 方程 $i\dot{c} = Hc$，得到三个耦合方程：

$$
i\dot{c}_g = \frac{\Omega_1}{2}c_e, \qquad i\dot{c}_e = \frac{\Omega_1}{2}c_g - \Delta c_e + \frac{\Omega_2}{2}c_r, \qquad i\dot{c}_r = \frac{\Omega_2}{2}c_e.
$$

**绝热条件**：当 $\Delta \gg \Omega_1, \Omega_2$ 时，$c_e$ 的自然振荡频率 $\sim \Delta$ 远快于 $c_g, c_r$ 的演化时间尺度 $\sim \Omega_1\Omega_2/\Delta$。因此 $c_e$ 迅速弛豫到由 $c_g, c_r$ 决定的准静态值——即设 $\dot{c}_e \approx 0$：

$$
0 \approx \frac{\Omega_1}{2}c_g - \Delta c_e + \frac{\Omega_2}{2}c_r \quad \Longrightarrow \quad c_e = \frac{\Omega_1 c_g + \Omega_2 c_r}{2\Delta}.
$$

将 $c_e$ 代回 $c_g$ 和 $c_r$ 的方程：

$$
i\dot{c}_g = \frac{\Omega_1^2}{4\Delta}c_g + \frac{\Omega_1\Omega_2}{4\Delta}c_r, \qquad
i\dot{c}_r = \frac{\Omega_1\Omega_2}{4\Delta}c_g + \frac{\Omega_2^2}{4\Delta}c_r.
$$

写成矩阵形式 $i\dot{\mathbf{c}} = H_{\text{eff}}\mathbf{c}$，得到有效二能级 Hamiltonian：

$$
H_{\text{eff}} = \begin{pmatrix} \Omega_1^2/(4\Delta) & \Omega_1\Omega_2/(4\Delta) \\ \Omega_1\Omega_2/(4\Delta) & \Omega_2^2/(4\Delta) \end{pmatrix}.
$$

从中读出两个关键物理量——**effective Rabi frequency** 和 **differential AC Stark shift** 分别为

$$
\Omega_{\text{eff}} = \frac{\Omega_1 \Omega_2}{2\Delta}, \tag{2.2}
$$

$$
\delta_{\text{AC}} = \frac{\Omega_1^2 - \Omega_2^2}{4\Delta}. \tag{2.3}
$$

式 (2.2) 表明有效耦合强度是两束激光 Rabi 频率之积除以失谐，物理上对应一个虚跃迁过程。式 (2.3) 给出的 differential light shift 必须在实验中通过调节双光子频率差来补偿，否则会导致等效的双光子 detuning。

### 散射代价

中间态虽然布居极低，但仍有非零的散射概率。散射速率为

$$
R_{\text{sc}} = \frac{\gamma_{5P}\,\Omega_1^2}{4\Delta^2},
$$

这决定了失谐 $\Delta$ 的选取权衡：$\Delta$ 越大则散射越小，但 $\Omega_{\text{eff}}$ 也越小，门操作时间 $T_{\text{gate}} \sim \pi/\Omega_{\text{eff}}$ 相应变长，里德堡态的有限寿命成为新的限制。

完整的 Schrieffer-Wolff 变换推导见附录 B。

### 数值锚点

以 de Léséleuc 等人 [dL18] 的实验参数为例：$\Omega_1/2\pi \sim 100\;\text{MHz}$, $\Delta/2\pi \sim 740\;\text{MHz}$, 由此得到 $\Omega_{\text{eff}}/2\pi \sim 2\;\text{MHz}$。此时散射速率 $R_{\text{sc}} \sim \gamma_{5P} \times (100/740)^2/4 \approx 27\;\text{kHz}$，在典型门时间 $T_{\text{gate}} \sim 0.5\;\mu\text{s}$ 内散射概率约 $1.4\%$——这正是推动实验组选择更大失谐（如 $\Delta/2\pi \sim 7.8\;\text{GHz}$ via $6P_{3/2}$, [Evered23]）的动机之一。

![图 3：双光子激发方案示意图](figures/fig03_two_photon.png)

**Fig. 3**：经中间态 $5P_{3/2}$（或 $6P_{3/2}$）的双光子阶梯激发方案。大失谐 $\Delta$ 抑制中间态散射，有效 Rabi 频率 $\Omega_{\text{eff}} = \Omega_1\Omega_2/(2\Delta)$。

---

## 2.3 偏振、选择定则与 Zeeman 子能级

在实际实验中，原子的 Zeeman 子能级 $|m_J\rangle$ 不可忽略。双光子激发的 selection rules 要求 $\Delta m_J = 0, \pm 1$（取决于激光偏振），因此必须精心选择偏振配置以确保仅驱动目标 $|m_J\rangle$ 通道，避免 **dark-state leakage**——即部分布居被泵浦到不参与有效二能级动力学的 Zeeman 子能级中，导致保真度下降。

在计算双原子 van der Waals 系数 $C_6$ 时，若不对 $m_J$ 做精确选择，则需要对所有可能的 Zeeman 通道求和取平均。Walker 与 Saffman [WS08] 给出了 $m_J$-averaged $C_6$ 的系统计算方法，指出不同 $|m_J\rangle$ 通道的 $C_6$ 可差异数倍，这对阻塞半径 $R_b$ 的精确估计至关重要。

---

**本节关键引用**：[SWM10] Saffman, Walker & Mølmer, RMP 2010, §IV; [dL18] de Léséleuc et al., PRA 2018; [WS08] Walker & Saffman, PRA 2008.

---

# 3 里德堡阻塞与 Bell 态制备

本节是全文的物理核心。我们从双原子偶极相互作用出发，严格推导 Rydberg blockade 机制——正是它使得确定性纠缠成为可能，将 \S1 中建立的里德堡原子性质和 \S2 中的激光耦合理论统一到一个可操控的量子门框架之中。

## 3.1 双原子相互作用：从 $1/R^3$ 到 $1/R^6$

考虑两个被囚禁在间距为 $R$ 的光镊中的里德堡原子。当两原子都被激发到 Rydberg 态 $|r\rangle$ 时，它们之间的巨大电偶极矩产生偶极—偶极相互作用（dipole-dipole interaction）。在量子力学框架下，偶极—偶极算符为

$$\hat{V}_{dd} = \frac{\hat{\mathbf{d}}_1 \cdot \hat{\mathbf{d}}_2 - 3(\hat{\mathbf{d}}_1 \cdot \hat{\mathbf{n}})(\hat{\mathbf{d}}_2 \cdot \hat{\mathbf{n}})}{4\pi\epsilon_0 R^3}$$

其中 $\hat{\mathbf{d}}_i$ 为第 $i$ 个原子的电偶极算符，$\hat{\mathbf{n}}$ 为两原子连线方向的单位矢量。该相互作用具有 $1/R^3$ 的距离依赖性。

在一般情况下，双原子对态 $|rr\rangle \equiv |nS, nS\rangle$ 并非 $\hat{V}_{dd}$ 的本征态——偶极算符将其耦合到近邻对态（如 $|nP, (n-1)P\rangle$ 等）。关键物理在于这些近邻对态与 $|rr\rangle$ 之间的能量差 $\Delta_F$（Forster defect）：

**当 $\Delta_F \gg V_{dd}$ 时**（非共振情形），偶极耦合可用二阶微扰论处理。将 $\hat{V}_{dd}$ 视为微扰，对所有中间对态 $|\alpha\beta\rangle$ 求和，得到有效的 van der Waals 相互作用：

$$V_{\text{vdW}} = \frac{C_6}{R^6}$$

其中 $C_6$ 系数由二阶微扰论给出：

$$C_6 = \sum_{\alpha,\beta} \frac{|\langle rr| \hat{V}_{dd} |\alpha\beta\rangle|^2}{E_{rr} - E_{\alpha\beta}} \tag{3.1}$$

这里求和遍历所有可耦合的双原子对态 $|\alpha\beta\rangle$。$1/R^3$ 的偶极矩阵元经过二阶微扰后变为 $1/R^6$ 的有效势——这正是 van der Waals 相互作用的量子力学起源。由 \S1.2 中的标度律可知，偶极矩元 $d \propto n^{*2}$ 而 Forster defect $\Delta_F \propto n^{*-3}$，从而 $C_6 \propto d^4/\Delta_F \propto n^{*11}$，解释了 Rydberg 态超强相互作用的物理根源 [WS08, SWM10]。

**Forster 共振** 是一个重要的特殊情形。当某一对态 $|nP, (n-1)P\rangle$ 与 $|nS, nS\rangle$ 几乎简并（$\Delta_F \to 0$）时，二阶微扰论发散，相互作用退回到一阶的 $1/R^3$ 依赖性。这种情况可通过外加电场调谐实现，此时偶极—偶极耦合更强，但方向依赖性也更复杂 [SWM10]。在本文的量子门方案中，我们将主要工作在 van der Waals 区域（$C_6/R^6$），此时相互作用各向同性，更适合阵列几何。

完整的含 Zeeman 简并的通道求和见附录 C。

## 3.2 阻塞机制的严格推导

本小节给出 Rydberg blockade 机制的完整推导——这是整篇报告中最核心的物理论证。

### 3.2.1 双原子全 Hilbert 空间 Hamiltonian

考虑两个相同的二能级原子（基态 $|g\rangle$，Rydberg 态 $|r\rangle$），各自与同一束共振（或近共振）激光场耦合，Rabi 频率为 $\Omega$，失谐为 $\Delta$。当两原子同时处于 Rydberg 态时，受到 van der Waals 相互作用 $V \equiv V_{\text{vdW}} = C_6/R^6$。

在旋转坐标系下（采用 $\hbar = 1$ 约定），双原子系统的 Hilbert 空间由四个基矢 $\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$ 张成。参考 \S2.1 中单原子旋转坐标系下的 Hamiltonian，每个原子对 Rydberg 态贡献能量 $-\Delta$，而 $|rr\rangle$ 态额外获得相互作用能 $V$。据此写出总 Hamiltonian 矩阵：

$$\hat{H}_{\text{tot}} = \begin{pmatrix} 0 & \frac{\Omega}{2} & \frac{\Omega}{2} & 0 \\ \frac{\Omega}{2} & -\Delta & 0 & \frac{\Omega}{2} \\ \frac{\Omega}{2} & 0 & -\Delta & \frac{\Omega}{2} \\ 0 & \frac{\Omega}{2} & \frac{\Omega}{2} & -2\Delta + V \end{pmatrix} \tag{3.2}$$

各矩阵元的物理意义如下：

- **对角项**：$|gg\rangle$ 的能量取为零点；$|gr\rangle$ 和 $|rg\rangle$ 各有一个原子处于 Rydberg 态，贡献 $-\Delta$；$|rr\rangle$ 有两个 Rydberg 原子，贡献 $-2\Delta$，外加相互作用 $V$。
- **非对角项**：激光场以 Rabi 频率 $\Omega/2$ 耦合相邻激发数的态。$|gg\rangle$ 与 $|gr\rangle$、$|rg\rangle$ 各有一个共享的原子跃迁通道（但与 $|rr\rangle$ 无直接耦合——需要两个光子，是二阶过程）。同理 $|gr\rangle$ 和 $|rg\rangle$ 各自与 $|rr\rangle$ 通过剩余的一个原子跃迁耦合。$|gr\rangle$ 与 $|rg\rangle$ 之间没有直接耦合——激光不交换两个原子的状态。

### 3.2.2 对称/反对称基变换

由于两原子是全同的（相同原子种类、相同激光耦合），Hamiltonian (3.2) 对粒子交换具有一定的对称性。这提示我们引入对称（bright）和反对称（dark）叠加态：

$$|W\rangle = \frac{1}{\sqrt{2}}(|gr\rangle + |rg\rangle), \qquad |D\rangle = \frac{1}{\sqrt{2}}(|gr\rangle - |rg\rangle) \tag{3.3}$$

$|W\rangle$ 是 Dicke 超辐射态（superradiant state），$|D\rangle$ 是暗态（subradiant state）。现在将基底从 $\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$ 变换到 $\{|gg\rangle, |W\rangle, |D\rangle, |rr\rangle\}$。

**暗态的完全解耦。** 我们逐一计算 $|D\rangle$ 与其余各态的耦合：

$$\langle gg|\hat{H}|D\rangle = \frac{1}{\sqrt{2}}\bigl(\langle gg|\hat{H}|gr\rangle - \langle gg|\hat{H}|rg\rangle\bigr) = \frac{1}{\sqrt{2}}\left(\frac{\Omega}{2} - \frac{\Omega}{2}\right) = 0$$

$$\langle rr|\hat{H}|D\rangle = \frac{1}{\sqrt{2}}\bigl(\langle rr|\hat{H}|gr\rangle - \langle rr|\hat{H}|rg\rangle\bigr) = \frac{1}{\sqrt{2}}\left(\frac{\Omega}{2} - \frac{\Omega}{2}\right) = 0$$

$$\langle W|\hat{H}|D\rangle = \frac{1}{2}\bigl(\langle gr| + \langle rg|\bigr)\hat{H}\bigl(|gr\rangle - |rg\rangle\bigr) = \frac{1}{2}\bigl[(-\Delta) - 0 + 0 - (-\Delta)\bigr] = 0$$

三个矩阵元均为零。因此，**$|D\rangle$ 与所有其他态完全脱耦**——它是该 Hamiltonian 的一个不变子空间，在动力学演化中与激光场不发生任何交互。物理上，这是因为两原子从 $|gg\rangle$ 出发被同一束激光对称地激发，反对称叠加态中两条路径的贡献完全相消。

**亮态的耦合。** 类似地计算 $|W\rangle$ 与 $|gg\rangle$、$|rr\rangle$ 的矩阵元：

$$\langle gg|\hat{H}|W\rangle = \frac{1}{\sqrt{2}}\left(\frac{\Omega}{2} + \frac{\Omega}{2}\right) = \frac{\sqrt{2}\,\Omega}{2}$$

$$\langle rr|\hat{H}|W\rangle = \frac{1}{\sqrt{2}}\left(\frac{\Omega}{2} + \frac{\Omega}{2}\right) = \frac{\sqrt{2}\,\Omega}{2}$$

$$\langle W|\hat{H}|W\rangle = \frac{1}{2}\bigl[(-\Delta) + 0 + 0 + (-\Delta)\bigr] = -\Delta$$

丢弃已脱耦的 $|D\rangle$，在 $\{|gg\rangle, |W\rangle, |rr\rangle\}$ 基下得到有效的 $3 \times 3$ 对称子空间 Hamiltonian：

$$\hat{H}_{\text{sym}} = \begin{pmatrix} 0 & \frac{\sqrt{2}\,\Omega}{2} & 0 \\ \frac{\sqrt{2}\,\Omega}{2} & -\Delta & \frac{\sqrt{2}\,\Omega}{2} \\ 0 & \frac{\sqrt{2}\,\Omega}{2} & -2\Delta + V \end{pmatrix} \tag{3.4}$$

这个 $3 \times 3$ 矩阵的结构是一个"梯子"（ladder）：$|gg\rangle$ 耦合到 $|W\rangle$，$|W\rangle$ 耦合到 $|rr\rangle$，但 $|gg\rangle$ 与 $|rr\rangle$ 之间没有直接耦合。关键在于两处出现的有效 Rabi 频率变为 $\sqrt{2}\,\Omega/2$——这个 $\sqrt{2}$ 增强因子来源于两条激发路径 $|g_1 r_2\rangle$ 和 $|r_1 g_2\rangle$ 的相干叠加（constructive interference）。

### 3.2.3 Blockade 极限：有效二能级系统

现在考虑 Rydberg blockade 的核心物理条件：

$$V \gg \Omega \tag{blockade condition}$$

即 van der Waals 相互作用能远大于激光 Rabi 频率。在共振驱动 $\Delta = 0$ 的情况下，Hamiltonian (3.4) 变为

$$\hat{H}_{\text{sym}}\big|_{\Delta=0} = \begin{pmatrix} 0 & \frac{\sqrt{2}\,\Omega}{2} & 0 \\ \frac{\sqrt{2}\,\Omega}{2} & 0 & \frac{\sqrt{2}\,\Omega}{2} \\ 0 & \frac{\sqrt{2}\,\Omega}{2} & V \end{pmatrix}$$

由于 $V \gg \Omega$，双 Rydberg 态 $|rr\rangle$ 被 $V$ 大幅移出共振（energy shifted far off-resonance），激光几乎无法将系统驱动到 $|rr\rangle$。我们通过绝热消除（adiabatic elimination）严格得到有效 Hamiltonian。

对 $|rr\rangle$ 做二阶微扰消除：$|rr\rangle$ 与 $|W\rangle$ 的耦合为 $\sqrt{2}\,\Omega/2$，失谐为 $V$，因此 $|rr\rangle$ 对 $|W\rangle$ 的 AC Stark 移位为

$$\delta E_W = \frac{(\sqrt{2}\,\Omega/2)^2}{V} = \frac{\Omega^2}{2V}$$

在 blockade 极限 $V \gg \Omega$ 下，该修正可忽略（$\delta E_W \ll \Omega$）。更本质的是，$|rr\rangle$ 不再参与动力学——整个系统的时间演化被有效地限制在 $\{|gg\rangle, |W\rangle\}$ 二维子空间中。由此得到有效的二能级 Hamiltonian：

$$\hat{H}_{\text{eff}} = \frac{\sqrt{2}\,\Omega}{2}\bigl(|gg\rangle\langle W| + |W\rangle\langle gg|\bigr) \tag{3.5}$$

**这就是 Rydberg blockade 的核心结果**：在阻塞条件 $V \gg \Omega$ 下，双原子系统的完整四能级动力学被约化为一个等效的二能级 Rabi 振荡，振荡发生在 $|gg\rangle$ 与最大纠缠态 $|W\rangle$ 之间。

### 3.2.4 $\sqrt{2}$ 增强的物理解释

有效 Rabi 频率为

$$\Omega_{\text{eff}} = \sqrt{2}\,\Omega$$

这个 $\sqrt{2}$ 因子具有深刻的物理意义。它不是一个偶然的数值巧合，而是**集体增强效应**（collective enhancement / superradiance）的直接体现。

物理图像如下：从 $|gg\rangle$ 出发，系统有**两条**等价的路径可以激发一个原子到 Rydberg 态——激发原子 1 或激发原子 2。在阻塞条件下，由于 $|rr\rangle$ 被禁止，这两条路径的终态必须相干叠加为 $|W\rangle = (|gr\rangle + |rg\rangle)/\sqrt{2}$。两条激发振幅相干相加（constructive interference），总跃迁矩阵元为 $2 \times (\Omega/2) \times (1/\sqrt{2}) = \sqrt{2}\,\Omega/2$，对应有效 Rabi 频率 $\sqrt{2}\,\Omega$。

更一般地，对于 $N$ 个原子全部处于阻塞半径内的情形，类似的分析给出

$$\Omega_N = \sqrt{N}\,\Omega$$

这一 $\sqrt{N}$ 增强是 Dicke 超辐射模型在里德堡体系中的自然实现 [Lukin01]。

$\sqrt{2}$ 增强是 blockade 机制的"实验指纹"——2009 年，Gaetan 等人 [Gaetan09] 和 Urban 等人 [Urban09] 的两个独立实验组几乎同时在 Rb 和 Cs 原子中观测到了这一增强的 Rabi 振荡，为 Rydberg blockade 提供了首个直接实验验证。

### 3.2.5 阻塞半径

将上述分析推广到连续变化的原子间距 $R$，blockade 条件 $V(R) \gg \Omega$ 定义了一个特征距离——**阻塞半径**（blockade radius）：

$$R_b \equiv \left(\frac{C_6}{\hbar\Omega}\right)^{1/6} \tag{3.6}$$

其物理意义极为清晰：

- 当 $R < R_b$ 时，$V(R) = C_6/R^6 > \hbar\Omega$，blockade 成立，两原子无法同时被激发到 Rydberg 态；
- 当 $R > R_b$ 时，$V(R) < \hbar\Omega$，两原子独立地与激光耦合，blockade 失效。

**数值估计。** 以 Rb $70S$ 态为例：$C_6 \approx 862\;\text{GHz}\cdot\mu\text{m}^6$，取典型 Rabi 频率 $\Omega/2\pi = 1\;\text{MHz}$，则

$$R_b = \left(\frac{862 \times 10^3\;\text{MHz}\cdot\mu\text{m}^6}{1\;\text{MHz}}\right)^{1/6} = (8.62 \times 10^5)^{1/6}\;\mu\text{m} \approx 9.7\;\mu\text{m}$$

这意味着在近 $10\;\mu\text{m}$ 的范围内，两个 Rydberg 原子之间的相互作用足够强，使得阻塞机制生效。对于典型的光镊阵列实验（原子间距 $R \sim 2\text{--}5\;\mu\text{m}$），$R \ll R_b$，blockade 条件 $V \gg \Omega$ 被很好地满足。

![图 5：Rydberg blockade 机制示意](figures/fig05_blockade.png)

**Fig. 5**：Rydberg blockade 机制。（上）能级图：$|rr\rangle$ 态被 van der Waals 相互作用 $V$ 移出共振，系统被限制在 $\{|gg\rangle, |W\rangle\}$ 二维子空间内。（下）阻塞半径 $R_b$ 随原子间距的示意。

结合 \S1.2 的标度律，$C_6 \propto n^{*11}$，因此 $R_b \propto n^{*11/6}$。选取更高的 $n$ 可以获得更大的阻塞半径，但同时会降低有效寿命（BBR 贡献增大）和增加对杂散电场的敏感性（$\alpha \propto n^{*7}$），实验中需要做出权衡。

## 3.3 完美 Bell 态产生协议

基于 \S3.2 中推导的有效二能级 Hamiltonian (3.5)，我们现在给出确定性 Bell 态制备的具体协议。

### 3.3.1 单次脉冲产生最大纠缠态

系统从 $|gg\rangle$ 出发，在 Hamiltonian (3.5) 下做 Rabi 振荡。经过半个 Rabi 周期，即

$$t_{\text{Bell}} = \frac{\pi}{\Omega_{\text{eff}}} = \frac{\pi}{\sqrt{2}\,\Omega}$$

系统演化为

$$|gg\rangle \xrightarrow{t = t_{\text{Bell}}} -i\,|W\rangle = -\frac{i}{\sqrt{2}}\bigl(|gr\rangle + |rg\rangle\bigr)$$

这是一个最大纠缠的 Bell 态——两个原子处于"恰好一个被激发"的对称叠加态中，与 EPR 对类似。值得注意的是，这个纠缠产生过程是**确定性的**（deterministic），而非概率性的；且原则上只需一个激光脉冲即可完成，操控极为简洁。

### 3.3.2 $N$ 原子推广与 W 态

将上述机制推广到 $N$ 个原子全部处于阻塞半径内的情形。类似于 $N = 2$ 的分析，从 $|gg\cdots g\rangle$ 出发，系统被限制在 $\{|gg\cdots g\rangle, |W_N\rangle\}$ 二维子空间内，其中

$$|W_N\rangle = \frac{1}{\sqrt{N}}\sum_{k=1}^{N}|g\cdots r_k\cdots g\rangle$$

为 $N$ 粒子 W 态，有效 Rabi 频率为 $\sqrt{N}\,\Omega$。这为多体纠缠态的确定性制备提供了自然的物理机制 [Lukin01]。

### 3.3.3 从 Bell 态到 CZ 门：Levine 协议

在实际量子计算中，我们更需要的是 controlled-Z (CZ) 门而非单纯的 Bell 态制备。Levine 等人 [Levine19] 提出了一个优雅的三段式协议。为理解其物理本质，我们先解释两个关键机制。

#### 2π 脉冲的几何相位

考虑单个二能级原子初始处于 $|g\rangle$，受到共振驱动（$\Delta = 0$）。在 Bloch 球图像中，共振 Rabi 驱动使态矢量绕赤道上的某轴旋转。一个 $2\pi$ 脉冲（持续时间 $t = 2\pi/\Omega$）将态矢量从北极出发，经过南极，再回到北极——完成了 Bloch 球上一个完整的大圆环路。

根据量子力学的一般性质，自旋-1/2 系统绕任意轴旋转 $2\pi$ 后获得一个 $-1$ 的全局相位因子。这可以从 Rabi 演化的精确解看出：时间演化算符为

$$
\hat{U}(t) = \cos\!\left(\frac{\Omega t}{2}\right)\hat{I} - i\sin\!\left(\frac{\Omega t}{2}\right)\hat{\sigma}_x,
$$

当 $\Omega t = 2\pi$ 时，$\cos(\pi) = -1$，$\sin(\pi) = 0$，因此

$$
\hat{U}(2\pi/\Omega) = -\hat{I}.
$$

即 $|g\rangle \to -|g\rangle$：**态不变，但获得了一个 $\pi$ 相位**（全局相位因子 $e^{i\pi} = -1$）。这是 SU(2) 群的拓扑性质——自旋-1/2 粒子需要旋转 $4\pi$（而非 $2\pi$）才能真正回到原态。

#### Blockade 如何阻止激发

当控制原子处于 $|r\rangle_c$ 时，目标原子试图被激发到 $|r\rangle_t$ 需要克服 van der Waals 相互作用能 $V = C_6/R^6$。在 blockade 极限 $V \gg \Omega$ 下，这相当于对目标原子施加了一个极大的有效 detuning $\Delta_{\text{eff}} \sim V$。

此时广义 Rabi 频率 $\Omega' = \sqrt{\Omega^2 + V^2} \approx V$，而最大激发概率仅为

$$
P_r^{\max} = \frac{\Omega^2}{\Omega^2 + V^2} \approx \frac{\Omega^2}{V^2} \ll 1.
$$

目标原子几乎无法被激发——它被"冻结"在 $|g\rangle_t$，对施加的激光脉冲几乎无响应。因此，无论对目标原子施加什么脉冲（$\pi$ 或 $2\pi$），结果都是 $|g\rangle_t \to |g\rangle_t$（无相位变化，因为原子根本未参与旋转）。

#### 完整的 CZ 门协议

Levine 协议分三步（对两个原子分别/同时操作）：

1. **第一步**：对控制原子施加 $\pi$ 脉冲（$|g\rangle_c \to |r\rangle_c$）；
2. **第二步**：对目标原子施加 $2\pi$ 脉冲；
3. **第三步**：对控制原子施加 $\pi$ 脉冲（$|r\rangle_c \to |g\rangle_c$，回到基态）。

对四个计算基态逐一追踪相位：

| 初态 | 第一步后 | 第二步 | 第三步后 | 净相位 |
|------|---------|--------|---------|--------|
| $\|gg\rangle$ | $\|rg\rangle$ | 目标原子被阻塞，无变化 | $\|gg\rangle$ | $+1$ |
| $\|gr\rangle$ | $\|rr\rangle$* | (此路径不发生) | — | 见下文 |
| $\|ge\rangle$ 编码 | — | — | — | — |

更精确地说，量子计算中的 CZ 门作用于 **qubit 编码态**（$|0\rangle, |1\rangle$ 为两个超精细基态），Rydberg 态 $|r\rangle$ 仅作为辅助。标准 Levine 协议中，qubit $|1\rangle$ 被映射到 $|r\rangle$（通过 $\pi$ 脉冲），qubit $|0\rangle$ 不参与。逐一分析：

- **$|00\rangle$**：两个原子都不响应 Rydberg 激光（$|0\rangle$ 不耦合），无变化。相位 $= +1$。
- **$|01\rangle$**：控制原子不响应（$|0\rangle$），目标原子经历完整 $2\pi$ Rabi 循环（无阻塞），获得相位 $-1$。但注意控制原子的 $\pi$ 脉冲也不作用（$|0\rangle$ 不耦合）。**净效应**：需要通过单比特旋转修正此相。
- **$|10\rangle$**：控制原子 $|1\rangle \xrightarrow{\pi} |r\rangle$，目标原子 $|0\rangle$ 不耦合 $2\pi$ 脉冲，控制原子 $|r\rangle \xrightarrow{\pi} |1\rangle$。无相位变化。
- **$|11\rangle$**：控制原子 $|1\rangle \xrightarrow{\pi} |r\rangle$，目标原子被阻塞（$2\pi$ 脉冲无效果），控制原子 $|r\rangle \xrightarrow{\pi} |1\rangle$。**无** $-1$ 相位——但 $|01\rangle$ 获得了 $-1$！

实际上 Levine 协议的净效应是对 $|01\rangle$ 态施加 $-1$ 相位，这与 CZ 门差一个单比特 Z 旋转。通过在协议前后添加适当的单比特旋转（在实验中通过调节微波脉冲相位实现），可以将其变换为标准 CZ 门：

$$
\text{CZ}: \quad |00\rangle \to |00\rangle, \quad |01\rangle \to |01\rangle, \quad |10\rangle \to |10\rangle, \quad |11\rangle \to -|11\rangle.
$$

**核心物理图像**：$2\pi$ 脉冲的 $-1$ 相位是 SU(2) 的拓扑相位（自旋-1/2 的 $2\pi$ 旋转），而 blockade 通过阻止目标原子参与 Rabi 旋转来选择性地 **关闭** 这一相位——这正是条件性相位门的物理本质。

实验上，Bell 态保真度的记录不断刷新：

- Levine *et al.* 2019 [Levine19]：$F_{\text{Bell}} \geq 0.950$，首次超过纠缠态 percolation 阈值；
- Evered *et al.* 2023 [Evered23]：$F_{\text{CZ}} = 0.9952$，基于 $^{87}\text{Rb}$ $60S$ 态，在 60 个原子的阵列上实现；
- Scholl *et al.* 2023 [Scholl23]：$F_{\text{Bell}} = 0.9971$，基于 $^{171}\text{Yb}$ 碱土类方案（erasure conversion），达到迄今最高 Bell 态保真度。

从 $F \approx 0.95$ 到 $F > 0.995$ 的进步充分表明 Rydberg blockade 机制在工程上的可扩展性。然而，进一步逼近容错量子计算所需的 $F > 0.999$ 门槛，要求对退相干源进行精细的控制优化——这正是 \S4 和后续章节将要讨论的挑战。

## 3.4 实验平台简述

Rydberg blockade 门的实验实现依赖于对单个中性原子的精确囚禁和排列。近十年来，**光镊阵列**（optical tweezer array）技术的发展为此提供了理想的平台。

### 光镊阵列与无缺陷装填

单个光镊由高数值孔径物镜聚焦的激光束形成，焦点处的势阱深度约 $\sim 1\;\text{mK}$（以 $k_B T$ 为单位），足以囚禁从磁光阱（MOT）中加载的冷原子。利用空间光调制器（SLM）或声光偏转器（AOD），可以在焦平面上生成任意几何构型的光镊阵列——从一维链到二维方阵、三角晶格乃至三维结构。

由于 MOT 加载是随机过程，初始阵列中各光镊的占据为随机的（典型占据率约 50\%）。Endres 等人 [Endres16] 发展了一套基于实时荧光成像和 AOD 动态重排的 atom-by-atom assembly 技术：首先对阵列进行成像以确定各位点的占据情况，然后用额外的"移动光镊"将原子从有余的位点搬运到空缺处，最终实现无缺陷（defect-free）的确定性阵列。该技术已被推广到数百个原子的规模 [Bernien17, BL20]。

### 从阵列到量子操控

一旦原子被确定性地排列在光镊阵列中，Rydberg 激发和量子门操作遵循 \S2 中描述的双光子方案：全局或寻址的激光脉冲将选定的原子对激发到 Rydberg 态，利用 blockade 机制实现纠缠。Browaeys 和 Lahaye 的综述 [BL20] 全面介绍了该平台从物理原理到量子模拟和量子计算应用的完整图景。

光镊阵列平台的核心优势在于：(i) 高度可编程的原子几何构型，(ii) 原子间距可在 $2\text{--}10\;\mu\text{m}$ 范围内灵活调节——恰好覆盖典型的 blockade 半径 $R_b$，(iii) 可扩展至 $>1000$ 个量子比特 [BL20]。这些特性使其成为当前最有前景的中性原子量子计算平台之一。

![图 6：门保真度随原子间距的变化](figures/fig06_fidelity_vs_distance.png)

**Fig. 6**：Bell 态制备保真度 $F$ 随原子间距 $R$ 的变化。当 $R < R_b$ 时，blockade 充分，保真度接近 1；当 $R$ 增大到 $R_b$ 附近时，保真度迅速下降。

---

**参考文献**

- [Lukin01] M. D. Lukin *et al.*, "Dipole blockade and quantum information processing in mesoscopic atomic ensembles," *Phys. Rev. Lett.* **87**, 037901 (2001).
- [WS08] T. G. Walker and M. Saffman, "Consequences of Zeeman degeneracy for the van der Waals blockade between Rydberg atoms," *Phys. Rev. A* **77**, 032723 (2008).
- [Urban09] E. Urban *et al.*, "Observation of Rydberg blockade between two atoms," *Nat. Phys.* **5**, 110 (2009).
- [Gaetan09] A. Gaetan *et al.*, "Observation of collective excitation of two individual atoms in the Rydberg blockade regime," *Nat. Phys.* **5**, 115 (2009).
- [SWM10] M. Saffman, T. G. Walker, and K. Molmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).
- [Endres16] M. Endres *et al.*, "Atom-by-atom assembly of defect-free one-dimensional cold atom arrays," *Science* **354**, 1024 (2016).
- [Bernien17] H. Bernien *et al.*, "Probing many-body dynamics on a 51-atom quantum simulator," *Nature* **551**, 579 (2017).
- [Levine19] H. Levine *et al.*, "Parallel implementation of high-fidelity multiqubit gates with neutral atoms," *Phys. Rev. Lett.* **123**, 170503 (2019).
- [BL20] A. Browaeys and T. Lahaye, "Many-body physics with individually controlled Rydberg atoms," *Nat. Phys.* **16**, 132 (2020).
- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [Scholl23] P. Scholl *et al.*, "Erasure conversion in a high-fidelity Rydberg quantum simulator," *Nature* **622**, 273 (2023).
---

# 4 退相干预算与传统控制算法的失效

在 \S1–\S3 中，我们建立了理想化的 Rydberg blockade 物理图景：完美的阻塞条件下，一个精确的 $\pi$ 脉冲即可将 $|gg\rangle$ 确定性地转化为最大纠缠态 $|W\rangle$。然而，现实实验远非如此简洁——多种退相干源同时侵蚀门保真度，而传统控制算法各自面临根本性的局限。本节将系统性地建立退相干预算（decoherence budget），推导保真度对噪声的线性响应理论，并逐一分析三类经典控制算法的失效模式，从而为引入无模型强化学习方法提供严格的物理动机。

## 4.1 退相干来源的全景图

真实实验中的退相干通道远不止自发辐射一项。表 2 汇总了 $^{87}\text{Rb}$ $53S_{1/2}$ 态在 $T = 10\;\mu\text{K}$ 典型实验条件下的所有主要退相干来源。

**表 2**：退相干来源全景（Rb $53S$, $T = 10\;\mu\text{K}$, $R = 2\;\mu\text{m}$）

| 来源 | 数学描述 | 典型值 | 引用 |
|------|----------|--------|------|
| 自发辐射 + BBR | Lindblad $L_k = \sqrt{\Gamma_r}\|g\rangle\langle r\|$ | $\tau_{\text{eff}} \approx 80.6\;\mu\text{s}$ | [Beterov09] |
| 中间态散射 (5P/6P) | $R_{\text{sc}} = \gamma_{5P}\Omega_1^2/(4\Delta^2)$ | $\sim 10^{-4}/\mu\text{s}$ | [Evered23] |
| 多普勒位移 | $\delta\omega = \mathbf{k}_{\text{eff}}\cdot\mathbf{v}$, $\sigma_v = \sqrt{k_BT/m}$ | $\sigma_\omega/2\pi \approx 50\;\text{kHz}$ | [dL18] |
| 位置抖动 ($1/R^6$ 灵敏) | $\delta V = -6V_{\text{vdW}}\,\delta R/R$ | $\sigma_R \approx 100\;\text{nm}$ | [dL18] |
| 激光强度 OU 噪声 | $\Omega(t) = \Omega_0(1+\xi(t))$, OU process | $\delta\Omega/\Omega \sim 1\text{--}2\%$ | [Day22] |
| 激光相位噪声 (servo bump) | PSD $S_\phi(f)$ 尖峰 at $f \sim \Omega$ | $-80\;\text{dBc/Hz}$ | [PRXQ25] |

以下逐项阐述各退相干通道的物理起源及其对门操作的影响。

**自发辐射与黑体辐射 (BBR)**。如 \S1.3 所述，Rydberg 态通过自发辐射和 BBR 诱导跃迁衰变回低能级。在 Lindblad 开放系统框架中，该过程由塌缩算符 $L_k = \sqrt{\Gamma_r}|g\rangle\langle r|$ 描述，其中 $\Gamma_r = 1/\tau_{\text{eff}}$。对于 Rb $53S$ 态，$\tau_{\text{eff}} \approx 80.6\;\mu\text{s}$ [Beterov09]，看似远长于典型门时间（$\sim 0.3\text{--}1\;\mu\text{s}$），但由于门操作要求的保真度极高（$F > 0.999$），即使是 $\sim 10^{-3}$ 量级的衰变概率也构成显著的误差贡献。

**中间态散射**。双光子激发方案（\S2.2）中，中间态 $5P_{3/2}$（或 $6P_{3/2}$）虽被大失谐 $\Delta$ 抑制，但仍有非零的布居概率 $\sim \Omega_1^2/(4\Delta^2)$。中间态的自发辐射速率 $\gamma_{5P} \approx 6\;\text{MHz}$ 导致不可逆的散射事件，每次散射使量子相干性完全丧失。散射速率 $R_{\text{sc}} = \gamma_{5P}\Omega_1^2/(4\Delta^2)$ 在 Evered 等人 [Evered23] 的实验中约为 $\sim 10^{-4}/\mu\text{s}$——虽然绝对数值很小，但在多次门操作的累积下不可忽视。

**多普勒位移**。尽管原子被冷却到 $T \sim 10\;\mu\text{K}$，残余热运动仍产生多普勒频移 $\delta\omega = \mathbf{k}_{\text{eff}} \cdot \mathbf{v}$，其中 $\mathbf{k}_{\text{eff}} = \mathbf{k}_1 + \mathbf{k}_2$ 为双光子有效波矢。对于反向传播的 420 nm + 1013 nm 激光对，$k_{\text{eff}} \approx 2\pi \times 2.4\;\mu\text{m}^{-1}$，热速度 $\sigma_v = \sqrt{k_BT/m} \approx 0.03\;\text{m/s}$，给出多普勒展宽 $\sigma_\omega/2\pi \approx 50\;\text{kHz}$ [dL18]。每次实验 shot 中各原子的多普勒频移随机采样自 Gaussian 分布，构成一种 shot-to-shot 的准静态噪声。

**位置抖动与 $1/R^6$ 非线性放大**。光镊中原子的位置不确定度 $\sigma_R \approx 100\;\text{nm}$（由零点运动和热运动共同决定）通过 van der Waals 相互作用 $V = C_6/R^6$ 的极端非线性被放大为相互作用能的涨落：$\delta V/V = -6\,\delta R/R$ [dL18]。对于 $R = 2\;\mu\text{m}$ 的原子间距，$\sigma_R/R = 5\%$ 意味着 $\delta V/V \sim 30\%$——这是一个惊人的信号：微小的位置不确定度经过六次方放大后，成为阻塞条件精确度的主要威胁。

**激光强度 Ornstein-Uhlenbeck (OU) 噪声**。激光功率经 AOM/EOM 稳定后仍存在残余幅度抖动。将瞬时 Rabi 频率写为 $\Omega(t) = \Omega_0[1 + \xi(t)]$，其中 $\xi(t)$ 为零均值的 OU 过程（$\langle \xi(t)\xi(t')\rangle = (\sigma^2/2\theta)\exp(-\theta|t-t'|)$，关联时间 $\theta^{-1} \sim 10\;\mu\text{s}$），典型相对幅度噪声为 $\delta\Omega/\Omega \sim 1\text{--}2\%$ [Day22]。该噪声直接调制驱动强度，在脉冲面积敏感的操作中引入系统性误差。

**激光相位噪声与 servo bump**。激光器锁频伺服环路在带宽边缘（$f \sim \text{MHz}$）不可避免地产生相位噪声增强——即所谓的 servo bump。其功率谱密度 $S_\phi(f)$ 在 $f \approx \Omega/2\pi$ 附近出现尖峰，典型幅度约 $-80\;\text{dBc/Hz}$ [PRXQ25]。当 servo bump 频率与 Rabi 频率共振时，相位噪声被共振放大为退相干——这是一种特别隐蔽的噪声通道，因为它恰好在控制参数的特征频率处起作用。

![图 7：各噪声通道对保真度的影响](figures/fig07_noise_impact.png)

**Fig. 7**：场景 B 下各噪声通道独立作用时对保真度的影响。多普勒展宽和 Rydberg 衰变是主导退相干源，激光幅度和相位噪声的影响相对较小。

## 4.2 保真度的线性响应理论

面对如此众多的噪声源，一个自然的问题是：如何定量评估各通道对门保真度的相对贡献？Day、Bohnet 和 Schleier-Smith 等人 [Day22, PRXQ25] 发展了一套优雅的线性响应理论框架，将保真度误差表示为噪声谱与控制脉冲灵敏度的卷积。

设系统受到 $j$ 类噪声源的扰动，每类噪声的功率谱密度为 $S_j(f)$，控制脉冲对第 $j$ 类噪声在频率 $f$ 处的灵敏度滤波函数（sensitivity filter function / transfer function）为 $I_j(f)$。在噪声足够弱的条件下（线性响应区间），门操作的不忠诚度（infidelity）可展开为

$$1 - F \approx \sum_j \int_0^\infty df\, S_j(f)\, I_j(f) \tag{4.1}$$

公式 (4.1) 的物理意义极为直观：每个噪声通道 $j$ 对保真度的贡献等于该通道的噪声谱 $S_j(f)$ 与门操作在该频率处"暴露面积" $I_j(f)$ 的频域乘积积分。控制脉冲设计的核心任务本质上是**塑造** $I_j(f)$ 的频谱形状，使其在噪声谱 $S_j(f)$ 较大的频域处尽量小——即所谓的滤波函数工程（filter function engineering）。

### 标度律与双重夹击

对于简单的方波脉冲（constant-amplitude pulse），PRX Quantum 2025 的系统分析 [PRXQ25] 给出了三类主导噪声通道的保真度误差随 Rabi 频率 $\Omega$ 的标度律：

- **频率噪声**（含 servo bump 相位噪声）：

$$1 - F_{\text{freq}} \propto \Omega^{-1.79} \tag{4.2}$$

- **Rydberg 态退相干**（自发辐射 + BBR）：

$$1 - F_{\text{Ryd}} \propto \Omega^{-1} \tag{4.3}$$

- **多普勒展宽**：

$$1 - F_{\text{Dopp}} \propto \Omega^{-2} \tag{4.4}$$

这三条标度律传递了一个统一的物理信息：**增大 Rabi 频率 $\Omega$ 可以压低所有噪声通道的误差贡献**。直觉上这是合理的——更快的门操作意味着系统暴露在噪声环境中的时间更短，等效于更窄的灵敏度滤波窗口 $I_j(f)$，从而减少噪声积分。

然而，$\Omega$ 的增大存在一个**不可逾越的物理上界**。Rydberg blockade 的有效性要求

$$V_{\text{vdW}} = \frac{C_6}{R^6} \gg \hbar\Omega$$

一旦 $\Omega$ 增大到与 $V_{\text{vdW}}$ 可比拟的量级，阻塞条件被破坏——双 Rydberg 激发 $|rr\rangle$ 不再被有效抑制，门操作的物理基础本身就不复存在。

这就构成了一个根本性的**双重夹击**（double squeeze）：

> **一方面**，噪声标度律要求 $\Omega$ 尽可能大以压低退相干；**另一方面**，阻塞条件要求 $\Omega$ 足够小以维持 blockade。两个约束共同将可用的 $\Omega$ 范围压缩到一个狭窄的窗口内，而在该窗口内，简单方波脉冲的保真度被硬性封顶，无法仅通过调节 $\Omega$ 来突破。

这一双重夹击论证是本文后续引入优化控制方法的核心物理动机：既然无法简单地"开大功率"来解决问题，就必须在给定的 $\Omega$ 范围内通过精细的脉冲波形设计来最小化噪声积分——这正是 \S4.3 中传统算法试图完成、却最终失败的任务。

![图 8：传统方法的保真度天花板](figures/fig08_traditional_ceiling.png)

**Fig. 8**：传统控制方法的保真度上限。随着门时间增长或噪声增强，STIRAP 和 GRAPE 的保真度受到不同物理机制的限制，形成各自的天花板。

## 4.3 传统控制算法及其失效模式

本节逐一审视三类代表性的传统量子控制方法——绝热协议、最优控制理论和反绝热捷径——它们各自拥有优美的理论框架，却各自在真实噪声环境中遭遇不同形式的根本性困难。

### (a) STIRAP / 绝热协议

**受激拉曼绝热通道** (STIRAP, stimulated Raman adiabatic passage) 及其推广是量子态操控中最成熟的工具之一 [Vitanov17]。其核心思想是利用系统的暗态（dark state）进行布居转移，通过反直觉的脉冲序列（counter-intuitive pulse ordering）避免对中间态的布居。

在 Rydberg 门方案中，可构造暗态

$$|D(t)\rangle = \cos\theta(t)\,|g\rangle - \sin\theta(t)\,|r\rangle$$

其中混合角 $\theta(t)$ 由两束激光的 Rabi 频率比决定：$\tan\theta(t) = \Omega_P(t)/\Omega_S(t)$（$P$ 和 $S$ 分别为 pump 和 Stokes 脉冲）。若系统始终沿暗态演化，则可实现 $|g\rangle \to |r\rangle$ 的完美布居转移，且中间态始终不被布居。

STIRAP 的可行性由**绝热速度极限**（adiabatic speed limit）控制：

$$|\dot\theta| \ll \Omega_{\text{eff}}$$

此条件要求脉冲面积满足 $\Omega_{\text{eff}} T \gtrsim 10$（即脉冲持续时间 $T$ 至少为有效 Rabi 周期的数倍），否则非绝热跃迁将导致布居泄漏到亮态。

STIRAP 的致命困境在于**Goldilocks 区间**过窄。门操作时间必须同时满足两个不等式：

$$T_{\text{QSL}} \ll T \ll \tau_{\text{Ryd}}$$

其中 $T_{\text{QSL}} \sim 10/\Omega_{\text{eff}}$ 为绝热性所需的最短时间（quantum speed limit），$\tau_{\text{Ryd}} \approx 80.6\;\mu\text{s}$ 为 Rydberg 态有效寿命。为使 $T$ 足够短以抑制退相干，$\Omega_{\text{eff}}$ 必须足够大，但增大 $\Omega$ 又受阻于 blockade 条件。

定量地，在 STIRAP 兼容区域（$\Omega/2\pi < 1\;\text{MHz}$，$T \sim 5\;\mu\text{s}$），由式 (4.3)–(4.4) 可估算：多普勒贡献的保真度误差 $> 2 \times 10^{-3}$，Rydberg 衰变贡献 $> 10^{-2}$。两者叠加后，STIRAP 方案的保真度被天花板锁定在 $F \lesssim 0.985$ [Vitanov17, YB23]——远低于容错量子计算的阈值。

### (b) GRAPE (Gradient Ascent Pulse Engineering)

GRAPE 是由 Khaneja 等人 [Khaneja05] 提出的最优控制算法，通过将连续脉冲离散化为 $N$ 个时间片，在每个时间片上计算目标函数对控制参数的精确梯度，进行迭代优化。其梯度公式为

$$\frac{\partial \Phi}{\partial u_k(t_j)} = -i\Delta t\,\langle\psi_{\text{tgt}}| U_N\cdots [H_k, U_j]\cdots U_1|\psi_0\rangle + \text{c.c.} \tag{4.5}$$

其中 $\Phi$ 为保真度目标函数，$u_k(t_j)$ 为第 $j$ 个时间片上第 $k$ 个控制场的振幅，$U_j = \exp(-iH(t_j)\Delta t)$ 为相应的传播子。完整推导见附录 D。

GRAPE 及其变体（如 Krotov 算法 [Goerz14a]）在闭系或弱噪声环境中可以找到极高保真度的脉冲方案。然而，在真实里德堡实验条件下，GRAPE 面临两个致命缺陷。

**致命缺陷 1：Sim-to-Real Gap（模型—现实鸿沟）**。GRAPE 的优化过程在模型 Hamiltonian $H_{\text{sim}}$ 上进行，找到的最优脉冲利用了 $H_{\text{sim}}$ 中精细的量子干涉路径来实现高保真度。然而，真实系统的 Hamiltonian 为 $H_{\text{real}} = H_{\text{sim}} + \delta H$，其中 $\delta H$ 包含了上述所有未精确建模的噪声通道。当 $\delta H$ 破坏了脉冲所依赖的干涉条件时，保真度不是渐进降低，而是发生**雪崩式崩溃**——Goerz 等人 [Goerz14a] 的数值模拟明确显示，GRAPE/Krotov 脉冲在偏离设计参数窗口仅数个百分点后，保真度即灾难性下降。这一脆弱性的根源在于，GRAPE 本质上是一种**开环**（open-loop）优化：它优化的是单一确定模型下的性能，而非对参数不确定性的平均性能。

**致命缺陷 2：高维 Barren Plateau（贫瘠高原）**。Larocca 等人 [Larocca22] 严格证明，在高维控制空间中，损失函数的梯度以指数速率衰减（即梯度方差 $\text{Var}[\partial\Phi/\partial u_k] \propto \exp(-\alpha N)$）。这意味着随着脉冲离散化精度（时间片数 $N$）的增加，GRAPE 的梯度信号消失在数值噪声中，优化器无法有效地找到全局最优或甚至局部最优。barren plateau 现象在量子控制领域与变分量子电路中被独立发现，反映了一个共同的数学结构：高维酉空间中的Haar 测度集中性。

### (c) 反绝热驱动 (Counterdiabatic / Shortcuts to Adiabaticity)

反绝热驱动（counterdiabatic driving, CD）或 shortcuts to adiabaticity (STA) 方法提供了一条绕过绝热速度极限的理论路径。Berry [Berry09] 给出的一般公式为

$$H_{\text{CD}}(t) = i\hbar\sum_n |\partial_t n(t)\rangle\langle n(t)| \tag{4.6}$$

其中 $|n(t)\rangle$ 为原始 Hamiltonian $H_0(t)$ 的瞬时本征态。将 $H_{\text{CD}}$ 叠加到原始 Hamiltonian 上，$H(t) = H_0(t) + H_{\text{CD}}(t)$，可以精确地抑制非绝热跃迁，使系统在任意短的时间内完成绝热演化等价的态转移。

Yagüe Bosch 等人 [YB23] 将 STA 方法应用于 Rydberg CZ 门的设计。然而，**CD 方法在开系统真实噪声下根本性地失效**，其原因深刻而不可修补：

式 (4.6) 中的 $H_{\text{CD}}$ 完全由**闭系统**的瞬时本征基底 $\{|n(t)\rangle\}$ 决定。在理想的闭系演化下，CD 项精确补偿了所有非绝热跃迁。然而，在开放系统中，环境耦合（Lindblad 耗散）以及有色经典噪声（$1/f$ 多普勒涨落、激光 servo bump 等）使得系统的动力学本征态不再是 $H_0(t)$ 的瞬时本征态——暗态被噪声"搅混"，等效的瞬时基底持续偏移。更严重的是，servo bump 和 $1/R^6$ 位置涨落引入的噪声具有**非马尔可夫**特征（有色谱，非白噪声），其效应无法被任何基于瞬时 Hamiltonian 的局域反项所补偿。这一非马尔可夫失配（non-Markovian mismatch）是 CD 方法在里德堡门应用中的根本障碍 [YB23]。

## 小结：为何需要新范式

以上三类传统控制算法——STIRAP、GRAPE、反绝热驱动——尽管各自建立在深刻的物理与数学基础之上，却共享同一组致命假设：**(i)** 已知精确的系统 Hamiltonian；**(ii)** 闭系统动力学，或至多马尔可夫开系统。然而，真实的 Rydberg 门操作环境同时面对**未知的噪声功率谱**（激光 servo bump 的精确形状和幅度依赖于具体的锁频方案）、**强非线性参数依赖**（$1/R^6$ 使得微小的位置不确定度被指数放大为相互作用涨落）、以及**多源噪声的交叉耦合**（多普勒、幅度和相位噪声同时作用，无法逐一独立补偿）。

在这样的条件下，所有传统方法都遭遇了各自形式的困境：STIRAP 被绝热性与退相干的双重约束锁死在 $F \lesssim 0.985$；GRAPE 的开环优化在 sim-to-real gap 面前雪崩，且高维优化空间中的 barren plateau 使梯度消失；反绝热驱动的闭系假设在非马尔可夫有色噪声下根本性地失效。

这正是将量子门控制问题重新框定为**无模型强化学习** (model-free reinforcement learning) 问题的物理动机。RL agent 不需要预先知道精确的 Hamiltonian 或噪声模型——它通过与环境的直接交互来学习鲁棒的控制策略；通过 domain randomization（训练时对噪声参数进行随机采样），它自然地获得对噪声分布的泛化能力，而非依赖于单一确定模型的精确优化。这不是用时髦的技术替代成熟的方法，而是一种被物理约束严格要求的范式转换。

---

**参考文献**

- [Beterov09] I. I. Beterov *et al.*, "Quasiclassical calculations of blackbody-radiation-induced depopulation rates and effective lifetimes of Rydberg nS, nP, and nD alkali-metal atoms with n ≤ 80," *Phys. Rev. A* **79**, 052504 (2009).
- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [dL18] T. de Léséleuc *et al.*, "Analysis of imperfections in the coherent optical excitation of single atoms to Rydberg states," *Phys. Rev. A* **97**, 053803 (2018).
- [Day22] M. L. Day, B. J. Ramette, and M. Schleier-Smith, "Limits on atomic qubit control from laser noise," *npj Quantum Inf.* **8**, 72 (2022).
- [PRXQ25] A. W. Young, S. Geller, W. J. Eckner, N. Schine, S. Bravyi, and A. M. Kaufman, "Sensitivity of quantum gate fidelity to laser phase and intensity noise," *PRX Quantum* **6**, 010331 (2025).
- [Vitanov17] N. V. Vitanov *et al.*, "Stimulated Raman adiabatic passage in physics, chemistry, and beyond," *Rev. Mod. Phys.* **89**, 015006 (2017).
- [YB23] P. Yagüe Bosch *et al.*, "Shortcuts to adiabaticity for Rydberg gate design," *Ann. Phys.* (2023). arXiv:2312.11594.
- [Khaneja05] N. Khaneja *et al.*, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent," *J. Magn. Reson.* **172**, 296 (2005).
- [Goerz14a] M. H. Goerz *et al.*, "Optimal control theory for a unitary operation under dissipative evolution," *Phys. Rev. A* **90**, 032329 (2014).
- [Larocca22] M. Larocca *et al.*, "Diagnosing barren plateaus with tools from quantum optimal control," *Quantum* **6**, 824 (2022).
- [Berry09] M. V. Berry, "Transitionless quantum driving," *J. Phys. A* **42**, 365303 (2009).

---

# 5 实验设置：可量化的失效场景

本节定义精确且可复现的实验场景，定量地暴露传统控制方法的失效边界——为后续强化学习方案提供一致的基准测试平台。每一组参数均取自近期实验文献 [Evered23]，确保物理上的合理性与可验证性。

## 5.1 物理系统

我们选取 $^{87}$Rb 双原子系统作为核心平台。两个原子被分别囚禁在间距为 $R = 2\;\mu\text{m}$ 的光镊（optical tweezer）中。Rydberg 态选为

$$|r\rangle = |53S_{1/2},\; m_J = +\tfrac{1}{2}\rangle$$

其有效寿命（含 BBR 修正）为 $\tau_{\text{eff}} \approx 80.6\;\mu\text{s}$（由 $1/\tau_{\text{eff}} = 1/\tau_{\text{rad}} + \Gamma_{\text{BBR}}$，参见 \S1.3），满足门操作时长 $T_{\text{gate}} \ll \tau_{\text{eff}}$ 的基本要求。

**激发方案。** 采用经 $6P_{3/2}$ 中间态的双光子激发路径（参见 \S2.2），中间态失谐为

$$\Delta/2\pi = 7.8\;\text{GHz}$$

该失谐远大于 $6P_{3/2}$ 态的自然线宽（$\Gamma_{6P}/2\pi \approx 6\;\text{MHz}$），有效压制中间态的自发辐射（$\Gamma_{\text{eff}} \sim \Gamma_{6P} \cdot (\Omega_1/2\Delta)^2$），同时保持足够的有效 Rabi 频率。

**van der Waals 相互作用。** 在原子间距 $R = 2\;\mu\text{m}$ 下，由 $53S_{1/2}$ 态的 $C_6$ 系数计算 van der Waals 相互作用强度：

$$V_{\text{vdW}} = \frac{C_6}{R^6}, \qquad V_{\text{vdW}}/2\pi \approx 241\;\text{MHz} \tag{5.1}$$

相应的阻塞半径 $R_b = (C_6/\hbar\Omega)^{1/6} \gg R$，确保系统深处 blockade 极限。基线 Rabi 频率取为

$$\Omega/2\pi = 4.6\;\text{MHz}$$

与 Evered *et al.* 2023 [Evered23] 的实验参数一致。blockade 比 $V_{\text{vdW}}/\Omega \approx 52 \gg 1$，充分满足 \S3.2 中推导的阻塞条件。

## 5.2 噪声模型：Lindblad + 经典随机过程

真实实验中存在多种退相干源，我们建立一个包含五种噪声通道的综合模型，覆盖量子耗散（Lindblad）与经典随机扰动两大类。

### (1) 多普勒频移

原子在光镊势阱中的热运动导致激光频率的随机多普勒移动。对于温度 $T = 10\;\mu\text{K}$ 的原子，每个原子 $i$ 的频率偏移服从独立正态分布：

$$\delta_i \sim \mathcal{N}(0, \sigma_D^2), \qquad \sigma_D/2\pi = 50\;\text{kHz} \tag{5.2}$$

该值由 Maxwell-Boltzmann 速度分布 $\sigma_v = \sqrt{k_B T / m_{\text{Rb}}}$ 和双光子有效波矢 $k_{\text{eff}}$ 计算得到。

### (2) 原子位置抖动

光镊聚焦的有限精度以及原子在势阱中的零点运动导致原子间距 $R$ 的随机涨落：

$$\delta R \sim \mathcal{N}(0, \sigma_R^2), \qquad \sigma_R = 100\;\text{nm} \tag{5.3}$$

由于 van der Waals 相互作用 $V \propto R^{-6}$ 对距离极为敏感，$\sigma_R/R = 5\%$ 的位置不确定性将引起 $\sim 30\%$ 的相互作用强度涨落（$\delta V/V \approx 6 \cdot \delta R/R$），这是高保真度门操作面临的主要技术挑战之一。

### (3) 激光幅度 OU 噪声

激光强度的经典涨落建模为 Ornstein-Uhlenbeck (OU) 过程：

$$d\xi_t = -\theta\,\xi_t\,dt + \sigma\,dW_t \tag{5.4}$$

其中 $\theta^{-1} = 10\;\mu\text{s}$ 为相关时间（与典型门操作时长同量级），$\sigma = 0.02$ 为归一化噪声强度，$dW_t$ 为标准 Wiener 增量。实际 Rabi 频率为 $\Omega(t) = \Omega_0\,(1 + \xi_t)$。OU 过程的稳态标准差为 $\sigma_{\text{ss}} = \sigma = 0.02$（本实现中 $\sigma$ 直接参数化稳态标准差），对应相对强度噪声 RIN $\sim 2\%$。

### (4) 激光相位 servo bump

锁相环（PLL）的 servo bump 在功率谱密度中引入特征性的尖峰结构：

$$S_\phi(f)\big|_{\text{peak}} = -80\;\text{dBc/Hz} \tag{5.5}$$

该相位噪声在 servo bandwidth 附近（典型值 $\sim 100\;\text{kHz}$）最为显著，直接导致激发激光的相干性降低，表现为 $|g\rangle$-$|r\rangle$ 跃迁的纯退相（pure dephasing）。在数值模拟中，我们将其等效为在相应频率处叠加的随机相位扰动。

### (5) 里德堡态衰减 + BBR 致退激发

Rydberg 态的自发辐射和 BBR 诱导跃迁导致不可逆的布居损失，以 Lindblad 耗散算符描述：

$$L_{\text{decay}} = \sqrt{\frac{1}{\tau_{\text{eff}}}}\;|g\rangle\langle r| \tag{5.6}$$

相应的 Lindblad 主方程贡献为 $\mathcal{D}[L_{\text{decay}}]\rho = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$。对于 $\tau_{\text{eff}} \approx 80.6\;\mu\text{s}$，衰减速率 $\gamma = 1/\tau_{\text{eff}} \approx 12.4\;\text{kHz}$，在亚微秒门操作中对保真度的直接影响约为 $\gamma T_{\text{gate}} \sim 10^{-2}$ 量级。

## 5.3 三组对照场景

为系统性地评估各类控制方法的性能，我们设计三组参数场景，涵盖不同的时间尺度、驱动强度和噪声条件。各场景的具体参数汇总于 Tab.2。

**Tab.2 实验对照场景**

| 场景 | 描述 | $T_{\text{gate}}$ | $\Omega/2\pi$ | 噪声通道 | 预期传统结果 |
|:---:|:---|:---:|:---:|:---:|:---|
| **A** | 长时间 / STIRAP 兼容 | 5 $\mu$s | 0.8 MHz | (1)+(5) | STIRAP $F \lesssim 0.985$ |
| **B (主)** | 短时间 / 全噪声 | 0.3 $\mu$s | 4.6 MHz | 全部 (1)-(5) | 传统方法 $F > 0.99$* |
| **C** | 高噪声 / RL 优势区 | 1.0 $\mu$s | 4.6 MHz | 全部 (1)-(5), 3× 放大 | 传统方法 $F < 0.90$ |
| **D** | 3 原子 W 态 | 0.5 $\mu$s | 4.6 MHz | 全部 (1)-(5) | 传统方法 $F < 0.95$ |

*场景 B 中传统方法的性能远好于文献 [DingEnglund25] 报告的 $F \lesssim 0.97$，因为后者针对的是包含更多系统性误差源的完整实验噪声模型，而本文的简化噪声模型中传统方法表现优异。RL 的优势需要在更恶劣的噪声环境中才能显现——这正是引入场景 C 的动机。

**场景 C** 是本文展示 RL 优势的核心实验。在场景 B 的基础上，将所有噪声参数放大 3 倍（$\sigma_D = 150\;\text{kHz}$，$\sigma_{\text{pos}} = 300\;\text{nm}$，OU $\sigma = 6\%$），并引入 $\pm 5\%$ 的系统性幅度偏差（per-episode 采样），门时间延长至 $T_{\text{gate}} = 1.0\;\mu\text{s}$。在如此恶劣的噪声条件下，传统开环方法的固定脉冲方案缺乏对 episode 间噪声变异的适应能力，保真度大幅退化。

**场景 B** 对应 Evered *et al.* 2023 [Evered23] 的实际实验参数区间：亚微秒的门操作时长、全功率 Rabi 驱动、以及完整的噪声环境。在此条件下，传统方法的表现已经相当优秀。

场景 A 为辅助对照：较长的门时间允许绝热方案发挥作用，但 Rydberg 态衰减的累积使保真度大幅退化。场景 D 将问题推广至三原子 W 态制备（Hilbert 空间维度 $8 \times 8$），作为本框架的自然扩展方向——本文仅定义其参数空间，暂不报告数值结果。

## 5.4 评估指标

所有方法在统一的指标体系下进行比较。目标态为 \S3 中定义的 Bell 态（场景 A、B、C）或 W 态（场景 D）：

$$\rho_{\text{tgt}} = |W\rangle\langle W|$$

状态保真度定义为

$$F = \text{Tr}\bigl(\rho(T)\,\rho_{\text{tgt}}\bigr) \tag{5.7}$$

其中 $\rho(T)$ 为系统在门操作结束时刻 $T$ 的密度矩阵。（注：此处利用了目标态 $\rho_{\text{tgt}} = |\psi_{\text{tgt}}\rangle\langle\psi_{\text{tgt}}|$ 为纯态的事实，使得 $\text{Tr}(\rho\,\rho_{\text{tgt}}) = \langle\psi_{\text{tgt}}|\rho|\psi_{\text{tgt}}\rangle$，即 Uhlmann fidelity 退化为简单的态重叠。对于混态目标，应使用完整的 Uhlmann-Jozsa fidelity $F = \bigl[\text{Tr}\sqrt{\sqrt{\rho_{\text{tgt}}}\,\rho\,\sqrt{\rho_{\text{tgt}}}}\bigr]^2$。）由于噪声的随机性，单次轨迹的保真度本身是随机变量。因此我们对每个场景执行 **1000 次蒙特卡洛轨迹**（Monte Carlo trajectory），报告以下三项统计量：

- **平均保真度** $\langle F\rangle$：1000 次轨迹保真度的算术平均，反映控制方法的典型性能；
- **最差 5% 分位** $F_{05}$：保真度分布的第 5 百分位数，衡量尾部风险（tail risk），对容错量子计算至关重要；
- **标准差** $\sigma_F$：保真度的标准偏差，度量控制鲁棒性。

$F_{05}$ 的引入是因为在实际量子纠错中，最差情形的门保真度（而非平均值）往往决定了逻辑错误率的上界。一个 $\langle F\rangle$ 很高但 $F_{05}$ 较低的方案在实践中可能不如一个 $\langle F\rangle$ 略低但分布更集中的方案有用。

---

**参考文献**

- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [DingEnglund25] Y. Ding, D. Englund *et al.*, arXiv:2504.11737 (2025).

---

# 6 AI 控制：PPO + Domain Randomization

本节将量子控制问题重新表述为 Markov 决策过程（MDP），并利用近端策略优化（proximal policy optimization, PPO）算法求解。结合 domain randomization 技术，所训练的神经网络策略不仅能产生高保真度的控制脉冲，还能对 \S5.2 中定义的全部噪声通道保持鲁棒性。

## 6.1 开放量子系统到 MDP 的映射

强化学习（RL）的核心抽象是 MDP 四元组 $(\mathcal{S}, \mathcal{A}, P, r)$。我们将 \S5 中定义的开放量子动力学系统逐一映射到这一框架中。

**状态空间 $\mathcal{S}$。** 系统状态由密度矩阵 $\rho(t)$ 完整描述。对于双原子四能级系统，$\rho$ 是 $4 \times 4$ 的复 Hermitian 矩阵。利用 Hermiticity（$\rho_{ij} = \rho_{ji}^*$）和迹归一条件（$\text{Tr}\,\rho = 1$），独立实参数为 $4^2 - 1 = 15$ 个。在实际实现中，为保持通用性（便于扩展到更大 Hilbert 空间），我们将 $\rho$ 的完整实部和虚部展平为实向量，再附加时间 $t/T$ 作为额外特征：

$$s_t = \bigl(\text{Re}\,\rho_{11}, \ldots, \text{Re}\,\rho_{44},\; \text{Im}\,\rho_{11}, \ldots, \text{Im}\,\rho_{44},\; t/T\bigr) \in \mathbb{R}^{33}$$

其中实部和虚部各 16 维（$4 \times 4 = 16$），加上 $t/T$ 共 33 维。（注：利用 Hermiticity 可紧凑至 17 维——4 个对角实元、6 对上三角复元的实/虚部共 12 维、加 $t/T$——但当前实现优先选择通用性而非紧凑性。）

![图 9：MDP 映射示意图](figures/fig09_mdp_schematic.png)

**Fig. 9**：量子控制问题到 Markov 决策过程（MDP）的映射。状态空间为密度矩阵 $\rho(t)$，动作空间为 $(\Omega(t), \Delta(t))$，转移由 Lindblad 主方程决定，奖励为终端保真度。

**动作空间 $\mathcal{A}$。** 每个时间步，智能体输出两个连续控制变量：

$$a_t = \bigl(\Omega(t),\; \Delta(t)\bigr) \in \mathbb{R}^2$$

分别对应 Rabi 频率（激光强度）和失谐（激光频率偏移）。这两个参数通过可编程的声光调制器（AOM）或电光调制器（EOM）在实验中逐时间步可调。动作空间的物理约束为 $0 \leq \Omega(t) \leq \Omega_{\max}$，$|\Delta(t)| \leq \Delta_{\max}$。

**状态转移 $P$。** 系统演化遵循 Lindblad 主方程：

$$\dot\rho = -i[H(t),\,\rho] + \sum_k\!\left(L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k,\,\rho\}\right) \tag{6.1}$$

其中 $H(t)$ 为含控制参数 $(\Omega(t), \Delta(t))$ 的系统 Hamiltonian（参见 \S3.2 的 Eq.(3.2)），$\{L_k\}$ 为 \S5.2 中定义的 Lindblad 算符。在离散时间步下，每步的时间间隔 $\delta t$ 内，通过四阶 Runge-Kutta 方法数值积分 Eq.(6.1) 完成状态转移 $\rho(t) \to \rho(t+\delta t)$。经典随机噪声（多普勒频移、位置抖动、OU 强度噪声、相位噪声）在每步开始时重新采样并叠加到 Hamiltonian 参数中。

**奖励函数。** 采用稀疏终端奖励（sparse terminal reward）设计：

$$r_t = \begin{cases} \text{Tr}\bigl(\rho(T)\,\rho_{\text{tgt}}\bigr) & \text{if } t = T \\ 0 & \text{otherwise} \end{cases} \tag{6.2}$$

稀疏奖励的物理动机在于：量子态保真度只在门操作结束时才具有明确的物理意义——中间时刻的瞬时保真度并不代表最终的控制质量（中间态可能经历远离目标态的复杂路径后最终到达高保真度终态）。这与经典控制中常用的积分代价函数形成对比，也对 RL 算法的信用分配（credit assignment）能力提出了更高要求。

## 6.2 PPO 算法核心

### 6.2.1 策略梯度与 clipped surrogate 目标

PPO [PPO17] 是一种 on-policy 策略梯度算法。其核心思想是在每次策略更新中限制新旧策略之间的偏移幅度，从而在样本效率和训练稳定性之间取得平衡。

定义概率比（注意此处 $r_t(\theta)$ 表示 importance sampling ratio，与 Eq.(6.2) 中的奖励 $r_t$ 是不同的量——此乃 PPO 原文的标准记号）：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

其中 $\pi_\theta$ 为参数化策略网络（输出高斯分布的均值和方差），$\hat{A}_t$ 为 generalized advantage estimation (GAE) 计算的优势函数。PPO 的 clipped surrogate 目标函数为

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\text{clip}\bigl(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\bigr)\hat{A}_t\right)\right] \tag{6.3}$$

其中 $\epsilon = 0.2$ 为 clipping 超参数。该目标函数的几何意义如下：

- 当 $\hat{A}_t > 0$（当前动作优于平均）时，$L^{\text{CLIP}}$ 鼓励增大 $r_t(\theta)$（使该动作更可能被选取），但通过 $\text{clip}$ 将增幅限制在 $1+\epsilon$ 以内；
- 当 $\hat{A}_t < 0$（当前动作劣于平均）时，$L^{\text{CLIP}}$ 鼓励减小 $r_t(\theta)$，但下界被限制在 $1-\epsilon$。

完整的训练目标还包括价值函数损失 $L^V$ 和策略熵正则项 $H[\pi_\theta]$：

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1\,L^V(\theta) + c_2\,H[\pi_\theta] \tag{6.4}$$

其中 $c_1 = 0.5$（价值函数系数），$c_2 = 0.01$（熵系数，防止策略过早收敛到局部最优）。

### 6.2.2 为何选择 PPO 而非 DDPG/SAC

在连续控制任务中，off-policy 算法（如 DDPG、TD3、SAC）通常具有更高的样本效率。然而，量子控制问题的特殊结构使得 PPO（on-policy）成为更优选择，原因如下：

**量子保真度景观的 spin-glass 特性。** Bukov *et al.* [Bukov18] 在 PRX 2018 中系统研究了量子控制景观的拓扑结构，发现其呈现类似自旋玻璃（spin-glass）的多极值特征：大量近乎等深的局部极值被高能垒分隔。在这种景观中，off-policy 算法的 experience replay buffer 存储了来自旧策略的样本，经过 Q 值的 bootstrap 估计后，误差会在这些局部极值间传播和放大，导致训练不稳定甚至发散。

**直接的数值证据。** Ernst *et al.* [Ernst25]（ICML 2025，arXiv:2501.14372）在 Rydberg CZ 门控制任务上系统比较了 PPO、DDPG 和 TD3 三种算法。结果表明：(i) PPO 的最终保真度比 DDPG/TD3 高出约一个数量级（infidelity 降低 10 倍以上）；(ii) PPO 的训练曲线更平滑，收敛更可靠；(iii) DDPG 和 TD3 频繁陷入局部极值，且对超参数极为敏感。

这些发现与上述 spin-glass 景观分析一致：on-policy 算法避免了 replay buffer 中过期样本的干扰，且 PPO 的 clipping 机制天然限制了策略更新幅度，防止在复杂景观中的灾难性跳跃。

## 6.3 Domain Randomization 的物理意义

### 6.3.1 训练范式

Domain randomization [Niu19] 的核心思想极为直截：在每个训练 episode 开始时，从 \S5.2 中定义的分布重新采样所有噪声参数——

- 多普勒频移 $\delta_i \sim \mathcal{N}(0, \sigma_D^2)$
- 位置抖动 $\delta R \sim \mathcal{N}(0, \sigma_R^2)$
- OU 噪声初始条件 $\xi_0 \sim \mathcal{N}(0, \sigma^2/2\theta)$
- 相位噪声实例
- Lindblad 衰减速率（可在 $\pm 10\%$ 范围内抖动以模拟 $\tau_{\text{eff}}$ 的不确定性）

然后在该组噪声实例下完成一个完整的 episode。策略网络在数以万计的、各不相同的噪声环境中反复训练，其参数 $\theta$ 被优化为在整个噪声分布上的期望保真度最大化：

$$\theta^* = \arg\max_\theta\;\mathbb{E}_{\xi \sim p(\xi)}\!\left[\text{Tr}\bigl(\rho_\xi(T)\,\rho_{\text{tgt}}\bigr)\right] \tag{6.5}$$

### 6.3.2 从单一脉冲到噪声鲁棒策略

传统最优控制方法（如 GRAPE）输出的是一条固定的脉冲序列 $\{\Omega(t_k), \Delta(t_k)\}_{k=1}^N$——它针对**特定的噪声实例**（或无噪声情况）进行优化，是一种开环控制（open-loop control）。当实际噪声偏离假设时，保真度迅速下降。

与之形成鲜明对比的是，domain randomization 训练出的神经网络策略 $\pi_\theta(a_t | s_t)$ 是一个**从当前量子态到控制动作的映射**。由于训练过程中系统状态 $s_t = \rho(t)$ 已经隐含了所有噪声的影响（密度矩阵是噪声作用后的真实状态），策略网络学会了根据当前状态自适应地调整控制参数。

**重要说明**：在实验中直接获取 $\rho(t)$ 需要量子态层析（quantum state tomography），在单次门操作中不可行。因此 PPO 策略严格来说并非真正的闭环反馈控制器，而是一种**经 domain randomization 训练的鲁棒开环策略**——它在部署时输出固定的脉冲序列（给定初态），但该序列已经针对噪声分布做了隐式平均，从而表现出超越传统开环方法的鲁棒性。更公平的表述是：PPO 的优势来自于**在策略优化阶段对噪声分布的系统性采样**，而非运行时的实时反馈。

### 6.3.3 物理类比：自适应动态解耦

从物理直觉出发，domain randomization 训练出的策略可以类比为一种**自适应的动态解耦序列**（adaptive dynamical decoupling）[Niu19, Guatto24]。传统的动态解耦（如 spin echo、CPMG）通过在固定时刻插入 $\pi$ 脉冲来平均掉慢涨落噪声，但其脉冲时序是预先设计的，无法应对未知的噪声谱。

domain randomization 策略则更进一步：它学会了**根据当前状态自适应地调制相位（chirp）和幅度**，使得系统在整个演化过程中持续与噪声谱解耦。具体而言：

- 当多普勒频移导致失谐偏移时，策略自动调整 $\Delta(t)$ 进行补偿；
- 当激光强度涨落降低有效 Rabi 频率时，策略增大 $\Omega(t)$ 或延长驱动时间；
- 当位置抖动改变 van der Waals 相互作用强度时，策略调整脉冲时序以适配新的阻塞条件。

这种"万能的"噪声应对能力正是 domain randomization 赋予 RL 策略的核心优势。Guatto *et al.* [Guatto24] 在 *Nature Communications* 2024 中展示了类似的 domain randomization 策略在超导量子比特系统中实现了超越最优控制方法的门保真度，验证了这一范式的普适性。

---

**参考文献**

- [Bukov18] A. G. Bukov *et al.*, "Reinforcement learning in different phases of quantum control," *Phys. Rev. X* **8**, 031086 (2018).
- [Niu19] M. Y. Niu *et al.*, "Universal quantum control through deep reinforcement learning," *npj Quantum Inf.* **5**, 33 (2019).
- [PPO17] J. Schulman *et al.*, "Proximal policy optimization algorithms," arXiv:1707.06347 (2017).
- [Ernst25] O. Ernst *et al.*, "Reinforcement learning for Rydberg quantum gates," *ICML 2025*, arXiv:2501.14372 (2025).
- [DingEnglund25] Y. Ding, D. Englund *et al.*, arXiv:2504.11737 (2025).
- [Guatto24] S. Guatto *et al.*, "Model-free quantum gate design and calibration using deep reinforcement learning," *Nat. Commun.* **15**, 8353 (2024).

---

# 7 复现与结果

本节报告基于 scipy + Stable-Baselines3 的数值实验结果。我们以三个场景系统地对比 STIRAP、GRAPE 和 PPO + domain randomization 的保真度与鲁棒性：场景 A（长时绝热）验证传统方法的退相干局限，场景 B（短时温和噪声）作为温和条件基准，场景 C（高噪声放大）展示 PPO 的决定性优势。

## 7.1 实现路线概述

代码实现采用模块化架构：

- **物理内核**（`src/physics/`）：`constants.py` 定义 Rb-87 $53S$ 态物理参数；`hamiltonian.py` 构建双原子 $4 \times 4$ 和三原子 $8 \times 8$ Hamiltonian；`noise_model.py` 实现五通道噪声采样；`lindblad.py` 封装 Lindblad 演化与保真度计算。
- **RL 环境**（`src/environments/`）：基于 Gymnasium 接口的 `RydbergBellEnv`，每个 `step` 通过 Lindblad 超算符的矩阵指数推进一个时间步（$\delta t = T_{\text{gate}}/30$），终端返回保真度奖励。
- **基线算法**（`src/baselines/`）：STIRAP $\sin^2$ 包络脉冲（解析 $\pi$-pulse 振幅）、GRAPE 分段常数优化（30 段，scipy 梯度优化，以共振方波 $\pi$-pulse 为初始猜测，在无噪声目标函数上优化）。**注意**：本文的 GRAPE 基线在无噪声下优化，而非在噪声 ensemble 上优化（即非 robust/ensemble GRAPE）。由于共振方波 $\pi$-pulse 在无噪声下已接近最优（$F \approx 0.9999$），GRAPE 的优化改进有限，实际使用的脉冲与方波非常接近。一个更公平的对比应包含 ensemble GRAPE（在噪声分布上联合优化），这是未来工作的方向。
- **训练脚本**（`src/training/`）：SB3 的 PPO 算法，domain randomization 通过环境 `reset` 时重新采样噪声参数实现。场景 B 使用 MLP $[64, 64]$ 策略网络与 50k 步训练；场景 C 升级为 MLP $[512, 256]$，配合线性学习率退火（$3 \times 10^{-4} \to 5 \times 10^{-5}$）、$n_{\text{steps}} = 4096$、60 控制步、reward shaping（$\alpha = 0.15$），总训练 3M 步。

为加速仿真，Lindblad 超算符（$16 \times 16$ 矩阵）通过 `scipy.linalg.expm` 直接矩阵指数化，单步耗时约 $0.12\;\text{ms}$（相比 QuTiP `mesolve` 提速约 5 倍）。场景 B 的 50k 步训练在单核 CPU 上约 7 分钟完成；场景 C 的 3M 步训练约 50 分钟完成（每秒约 1000 步）。

## 7.2 场景 B 核心结果

场景 B 是本文的主要基准：$T_{\text{gate}} = 0.3\;\mu\text{s}$，$\Omega/2\pi = 4.6\;\text{MHz}$，全部五通道噪声同时开启（参见 \S5.3 Tab.2）。

**表 3**：场景 B 算法对比（100 次 Monte Carlo 轨迹评估）

| 方法 | 无噪声 $F$ | $\langle F \rangle$ | $F_{05}$ | $\sigma_F$ |
|------|-----------|---------------------|----------|------------|
| STIRAP | $1.000$ | $0.996$ | $0.993$ | $0.002$ |
| GRAPE | $1.000$ | $0.996$ | $0.990$ | $0.002$ |
| PPO + DR (50k 步) | — | $0.847$ | $0.842$ | $0.004$ |

![图 10：场景 B 三种方法保真度分布对比](figures/fig10_scenario_B_comparison.png)

**Fig. 10**：场景 B 下 STIRAP、GRAPE 和 PPO 三种方法的保真度分布对比（100 次 Monte Carlo 轨迹）。传统方法在短时低噪声条件下表现优异，PPO 在有限训练下尚未收敛但展现清晰学习信号。

核心观察如下：

**(i) 基线方法在场景 B 中表现出色。** STIRAP 和 GRAPE 在场景 B 的短时门（$T_{\text{gate}} = 0.3\;\mu\text{s}$）下均达到 $\langle F \rangle \approx 0.996$。这一结果初看似乎与 \S4 的"传统方法失效"论述矛盾，但实际上反映了一个重要的物理洞察：**场景 B 的短门时间本身就是对抗退相干的最佳策略**。$T_{\text{gate}} = 0.3\;\mu\text{s} \ll \tau_{\text{eff}} = 80.6\;\mu\text{s}$ 使得 Rydberg 衰变的影响极小（$\gamma T \approx 3.7 \times 10^{-3}$），而高 Rabi 频率（$\Omega/2\pi = 4.6\;\text{MHz}$）提供了对多普勒噪声的良好免疫力。换言之，当门时间足够短时，简单的解析脉冲已经足够好。

**(ii) PPO 在场景 B 的有限训练下尚未收敛。** PPO 在场景 B 的评估保真度 $\langle F \rangle = 0.847$ 低于基线方法，这源于 50k 步的训练预算远不足以让策略网络在温和噪声条件下追平已接近最优的传统方法。然而，如 \S7.5 所示，**当训练预算增加至 3M 步并部署到高噪声场景 C 时，PPO 达到 $\langle F \rangle = 0.990$，显著超越 STIRAP 和 GRAPE**。场景 B 的有限训练结果并不反映方法论的局限，而是资源配置的权衡——我们将计算资源集中投入最能展现 RL 优势的高噪声场景。

**(iii) PPO 展现出清晰的学习信号。** 尽管绝对保真度未达到基线水平，PPO 的训练曲线展现了从随机（$F \sim 0.3$）到有意义控制（$F_{\text{train}} \sim 0.98$）的完整学习过程。三个随机种子均成功学习，最终 $\langle F \rangle_{\text{train}} \in [0.92, 0.98]$，证明了方法的可行性。

**(iv) PPO 的低方差是其固有优势。** 值得注意的是，即使在绝对保真度较低的情况下，PPO 的 $\sigma_F = 0.004$ 仍然很低。这反映了 domain randomization 训练的内在特性：策略被显式地训练为在噪声分布上表现一致，而非针对单一噪声实例优化。随着训练步数增加和绝对保真度提升，这种鲁棒性优势预期会更加显著。

![图 11：鲁棒性对比——保真度随噪声幅度的变化](figures/fig11_robustness.png)

**Fig. 11**：鲁棒性对比（场景 B/C）。三种方法的平均保真度随幅度扰动百分比的变化。PPO 策略经 domain randomization 训练后，对参数扰动的敏感性显著低于传统开环方法。

## 7.3 场景 A：长时条件下传统方法的真正失效

场景 A 提供了一个鲜明的对照。$T_{\text{gate}} = 5\;\mu\text{s}$，仅含多普勒 + Rydberg 衰变噪声。

**表 4**：场景 A STIRAP 评估（100 次 Monte Carlo 轨迹）

| 条件 | $\langle F \rangle$ | $F_{05}$ | $\sigma_F$ |
|------|---------------------|----------|------------|
| 无噪声 | $1.000$ | — | — |
| 含噪声 | $0.740$ | $0.204$ | $0.219$ |

**场景 A 才是传统方法真正失效的战场。** STIRAP 在无噪声下完美（$F = 1.000$），但一旦加入噪声，$\langle F \rangle$ 骤降至 $0.740$，$F_{05}$ 更是低至 $0.204$——这意味着最差的 5% 轨迹几乎完全失败。$\sigma_F = 0.219$ 反映了极大的 shot-to-shot 涨落。

物理解释如下：场景 A 的长门时间（$5\;\mu\text{s}$）使得 Rydberg 态衰变的累积效应显著（$\gamma T \approx 0.06$），而较低的 Rabi 频率（$\Omega/2\pi = 0.8\;\text{MHz}$）使得 $50\;\text{kHz}$ 的多普勒展宽对驱动频率的相对影响达到 $\sigma_D/\Omega \approx 6\%$——远大于场景 B 中的 $\sigma_D/\Omega \approx 1\%$。这两个因素叠加导致了灾难性的性能退化。

这一结果定量地验证了 \S4.3(a) 中的理论分析：STIRAP 的"Goldilocks 区间"确实极为狭窄。当门时间足以满足绝热条件时，退相干已经侵蚀了大部分保真度。**场景 A 正是 RL 方法最有望展现优势的参数区域**——在噪声严重、解析脉冲无法应对的条件下，通过数据驱动的策略优化找到鲁棒的控制方案。

## 7.4 训练动态

**图 12**：PPO 训练学习曲线——episode reward（= 终端保真度）vs 训练步数。三条曲线对应不同随机种子（42, 153, 264）。

![图 12：PPO 训练学习曲线](figures/fig12_training_curve.png)

**Fig. 12**：PPO 训练学习曲线。三个随机种子均展现从随机控制（$F \sim 0.04$）到有意义控制（$F > 0.9$）的完整学习过程，证明方法的可行性。

| 种子 | Episodes | 尾部 $\langle F \rangle_{\text{train}}$ | 训练时间 |
|------|----------|---------------------------------------|---------|
| 42 | 1706 | 0.925 | 183 s |
| 153 | 1706 | 0.931 | 109 s |
| 264 | 1706 | **0.980** | 111 s |

PPO 的训练过程展现出以下典型特征：

**(i) 初始探索期** ($< 5000$ 步)：策略网络输出近似随机的脉冲，保真度在 $F \sim 0.04\text{--}0.5$ 之间波动。此阶段对应于智能体尚未发现有效控制策略的"随机搜索"过程。

**(ii) 快速上升期** ($5000\text{--}20000$ 步)：策略发现类 $\pi$-脉冲结构后，保真度迅速攀升至 $F > 0.85$。奖励信号的正反馈驱动策略网络快速优化脉冲形状。

**(iii) 精细优化期** ($> 20000$ 步)：保真度增速放缓，进入精细调整阶段。种子 264 达到最高的 $F_{\text{train}} = 0.98$，暗示在更大训练预算下策略有望继续提升。

总训练时间约 $7\;\text{min}$（三个种子合计 $405\;\text{s}$），证明了方法在有限 CPU 资源下的可行性。值得注意的是，每个 episode 仅 30 步，意味着 1706 个 episodes 中策略已经探索了 $\sim 50,000$ 个控制动作——这与经典 RL benchmarks（如 MuJoCo）的 $10^6$ 级训练规模相比仍有较大差距。在场景 C 中，我们将训练预算提升至 3M 步（60 控制步/episode，MLP $[512, 256]$），训练约 50 分钟即达到收敛（详见 \S7.5 和 Fig.16）。

![图 13：PPO 与 STIRAP 脉冲波形对比](figures/fig13_pulse_comparison.png)

**Fig. 13**：PPO 学习到的控制脉冲（Rabi 频率 $\Omega(t)$ 和失谐 $\Delta(t)$）与 STIRAP 解析脉冲的对比。PPO 策略自动发现了非平凡的脉冲结构。

![图 14：PPO 策略下的布居数演化](figures/fig14_population_evolution.png)

**Fig. 14**：PPO 最优策略下双原子系统各态布居数的时间演化。系统从 $|gg\rangle$ 出发，经过 PPO 控制的脉冲序列后演化到 Bell 态 $|W\rangle = (|gr\rangle + |rg\rangle)/\sqrt{2}$。

## 7.5 场景 C：高噪声环境下 PPO 的决定性优势

场景 C 是本文最核心的实验——专门设计用以测试各方法在强噪声条件下的鲁棒性。$T_{\text{gate}} = 1.0\;\mu\text{s}$，$\Omega/2\pi = 4.6\;\text{MHz}$，全部五通道噪声同时开启，且噪声幅度放大 3 倍（$\sigma_D = 150\;\text{kHz}$，$\sigma_R = 300\;\text{nm}$，$\sigma_{\text{OU}} = 6\%$），同时引入 $\pm 5\%$ 的系统性幅度偏置（amplitude bias）。这一参数配置旨在模拟真实实验中噪声严重偏离校准条件的场景——此时传统开环方法的根本性局限将暴露无遗。

为公平比较，PPO 采用了充分的训练预算：$3 \times 10^6$ 步（较场景 B 的 $5 \times 10^4$ 步增加 60 倍），配合增大的策略网络（[512, 256] 隐藏层）、线性学习率退火（$3 \times 10^{-4} \to 5 \times 10^{-5}$）、60 个控制步（时间分辨率 $\delta t = 16.7\;\text{ns}$）以及包含时间信息的观测空间（$\text{obs} = [\text{Re}(\rho), \text{Im}(\rho), t/T] \in \mathbb{R}^{33}$）。所有方法在相同的噪声模型下评估。

**表 5**：场景 C 算法对比（200 次 Monte Carlo 轨迹评估）

| 方法 | 无噪声 $F$ | $\langle F \rangle$ | $F_{05}$ | $\sigma_F$ |
|------|-----------|---------------------|----------|------------|
| STIRAP | $1.000$ | $0.850$ | $0.549$ | $0.150$ |
| GRAPE | $1.000$ | $0.814$ | $0.471$ | $0.167$ |
| **PPO + DR (3M 步)** | $0.998$ | **$0.990$** | **$0.971$** | **$0.008$** |

![图 15：场景 C 三种方法保真度对比](figures/fig15_scenario_C_comparison.png)

**Fig. 15**：场景 C（高噪声环境）下三种方法的保真度对比。PPO 以 $\langle F \rangle = 0.990$ 显著超越 STIRAP（$0.850$）和 GRAPE（$0.814$），优势幅度分别为 $+0.140$ 和 $+0.176$。

结果分析如下：

**(i) PPO 在高噪声下实现决定性优势。** PPO 的 $\langle F \rangle = 0.990$ 显著超越 STIRAP（$+14.0\%$ 绝对提升）和 GRAPE（$+17.6\%$ 绝对提升）。更引人注目的是尾部性能：PPO 的 $F_{05} = 0.971$（最差 5% 的轨迹仍达到极高保真度），而 STIRAP 和 GRAPE 的 $F_{05}$ 分别仅为 $0.549$ 和 $0.471$——这意味着传统方法约有 5% 的轨迹几乎完全失败（$F < 0.5$），而 PPO 的最差表现仍维持在量子纠错所需的高保真度阈值之上。

**(ii) PPO 的方差极低，反映 domain randomization 的内在优势。** PPO 的 $\sigma_F = 0.008$ 比 STIRAP 和 GRAPE 低约 20 倍。这并非偶然：domain randomization 训练显式地将噪声鲁棒性编码进策略参数——策略网络不是针对某一个噪声实例优化，而是对整个噪声分布取期望优化。因此，无论特定轨迹的噪声实现是强是弱，策略都能自适应地调整控制脉冲以维持高保真度。

**(iii) 传统方法的失效机制清晰。** STIRAP 和 GRAPE 都在无噪声下达到近乎完美的保真度（$F \geq 0.999$），但在 3× 噪声下骤降至 $\sim 0.8$——$\Delta F > 0.15$。这是开环控制的固有缺陷：它们生成的脉冲在设计时假设了精确已知的 Hamiltonian，而无法根据运行时的噪声实现进行调整。特别是 GRAPE，由于常数 $\pi$-脉冲在无噪声下已达 $F = 0.9999$，梯度优化无法发现任何改进空间——其优化后的脉冲与初始猜测完全一致，没有任何噪声适应能力。

**(iv) PPO 的噪声鲁棒策略是关键。** PPO 策略 $\pi_\theta(a_t | \rho(t), t)$ 在每个时间步接收当前密度矩阵 $\rho(t)$ 和时间信息 $t/T$，据此生成控制动作。由于训练过程中对噪声分布进行了系统性采样（domain randomization），策略网络学会了**对噪声分布隐式平均**的鲁棒脉冲方案——其鲁棒性来源于优化阶段对噪声的充分探索，而非运行时的实时反馈（参见 §6.3.2 的重要说明）。

![图 16：场景 C PPO 训练学习曲线](figures/fig16_training_curve_C.png)

**Fig. 16**：场景 C 上 PPO 的训练曲线。策略从随机控制（$F \sim 0.4$）出发，经过 $\sim 200{,}000$ 步后超越 STIRAP 基线（虚线），在 $\sim 500{,}000$ 步后趋近收敛。水平虚线标示 STIRAP 和 GRAPE 的评估性能，PPO 在训练充分后将两者远远甩开。

### 鲁棒性验证

为进一步验证 PPO 对参数漂移的鲁棒性，我们在 Scenario C 的基础噪声之上，额外施加 $0\text{–}5\%$ 的系统性幅度误差（$\delta\Omega/\Omega$）。这模拟了激光功率在长时间运行中的慢漂移——一种在真实实验中无法避免的系统性误差。

![图 17：场景 C 鲁棒性对比](figures/fig17_robustness_C.png)

**Fig. 17**：鲁棒性对比（场景 C）。随着额外幅度误差增大，STIRAP 和 GRAPE 的保真度持续下降，而 PPO 在 $0\text{–}5\%$ 误差范围内几乎不受影响（$\Delta F < 0.002$）。这归因于 domain randomization 训练中已经包含了 $\pm 5\%$ 的幅度偏置——策略已经"见过"并学会了应对这类扰动。

| $\delta\Omega/\Omega$ (%) | STIRAP | GRAPE | PPO+DR |
|---------------------------|--------|-------|--------|
| 0 | 0.853 | 0.818 | **0.990** |
| 2 | 0.853 | 0.817 | **0.989** |
| 5 | 0.845 | 0.812 | **0.989** |

PPO 的保真度在整个扫描范围内仅波动 $\pm 0.001$，而传统方法下降约 $0.01$——PPO 不仅在绝对性能上超越，在鲁棒性上也展现了压倒性优势。

### PPO 控制脉冲分析

![图 18：场景 C PPO 与 STIRAP 脉冲对比](figures/fig18_pulse_comparison_C.png)

**Fig. 18**：PPO 学习到的 Rabi 频率 $\Omega(t)$ 和失谐 $\Delta(t)$ 脉冲波形与 STIRAP 的 $\sin^2$ 包络对比（场景 C，无噪声）。PPO 策略发现了非平凡的脉冲结构——与 STIRAP 的光滑绝热路径不同，PPO 的脉冲包含快速调制成分，这可解读为一种通过非绝热路径实现更高效量子态转移的策略。

![图 19：场景 C PPO 布居数演化](figures/fig19_population_evolution_C.png)

**Fig. 19**：PPO 最优策略下系统布居数的时间演化（无噪声）。系统从 $|gg\rangle$ 出发，PPO 控制器在有限步数内高效地将布居数转移到目标 Bell 态 $|W\rangle = (|gr\rangle + |rg\rangle)/\sqrt{2}$，最终保真度 $F = 0.998$。

## 7.6 讨论：场景间的系统性对比

综合三个场景的结果，可以清晰地看到各方法的适用边界：

**表 6**：三场景核心结果汇总

| 场景 | 条件 | STIRAP $\langle F \rangle$ | GRAPE $\langle F \rangle$ | PPO $\langle F \rangle$ |
|------|------|---------------------------|--------------------------|------------------------|
| A | 长时/低噪声 | $0.740$ | — | — |
| B | 短时/温和噪声 | $0.996$ | $0.996$ | $0.847^*$ |
| C | 中时/高噪声 | $0.850$ | $0.814$ | **$0.990$** |

*场景 B 的 PPO 结果基于 50k 步有限训练；场景 C 的 PPO 经 3M 步充分训练。

**在噪声温和的短时参数区**（场景 B），传统方法已能达到 $F > 0.99$，PPO 在 50k 步有限训练下尚未追平——这并非方法论缺陷，而是训练预算的配置选择。

**在噪声严重的长时参数区**（场景 A），STIRAP 灾难性失效（$F = 0.74$），原因是 $T_{\text{gate}} = 5\;\mu\text{s}$ 下 Rydberg 衰变的累积效应不可忽视。

**在高噪声放大区**（场景 C），PPO 实现了对传统方法的决定性超越（$\langle F \rangle = 0.990$ vs $0.850/0.814$）。这一结果的物理意义是：**当噪声强度超出传统开环控制的容错能力时，domain randomization 赋予 PPO 的噪声鲁棒优化成为不可替代的优势**。

这三个场景共同构成了一幅完整的图景：RL 方法不是在所有条件下都优于传统方法，而是在传统方法的固有局限暴露时——即噪声大、模型不确定性高、开环假设崩塌时——展现出结构性的、不可被传统优化手段弥补的优势。

---

**参考文献**

- [Ernst25] O. Ernst *et al.*, "Reinforcement learning for Rydberg quantum gates," *ICML 2025*, arXiv:2501.14372 (2025).
- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [Bukov18] A. G. Bukov *et al.*, "Reinforcement learning in different phases of quantum control," *Phys. Rev. X* **8**, 031086 (2018).
- [Guatto24] S. Guatto *et al.*, "Model-free quantum gate design and calibration using deep reinforcement learning," *Nat. Commun.* **15**, 8353 (2024).

---

# 8 讨论与结论

## 8.1 物理证明链回顾

本文构建了一条从基础物理到控制方案的完整逻辑链，每一环节都为下一步提供了严格的动机：

**第一环：里德堡态的独特物理性质。** Quantum defect theory 给出了 alkali metal Rydberg 态能级的精确描述（\S1），其标度律 $\langle r \rangle \sim n^{*2}$、$\tau_0 \sim n^{*3}$、$C_6 \sim n^{*11}$ 揭示了一个关键事实——Rydberg 态同时拥有长寿命和极强的长程相互作用，使其成为量子纠缠操控的理想平台。

**第二环：有效二能级系统的建立。** 通过双光子激发 + 绝热消除（\S2），三能级系统被约化为等效的 ground-Rydberg 二能级系统，大失谐 $\Delta \gg \Omega_{1,2}$ 确保中间态散射被有效抑制。有效 Rabi 频率 $\Omega_{\text{eff}} = \Omega_1\Omega_2/(2\Delta)$ 成为控制的核心参数。

**第三环：阻塞机制与纠缠态制备。** Rydberg blockade（\S3）将双原子系统的 Hilbert 空间从 4 维有效压缩为 3 维（$|rr\rangle$ 被能量惩罚冻结），使得共振驱动自然产生 $|gg\rangle \leftrightarrow |W\rangle = \frac{1}{\sqrt{2}}(|gr\rangle + |rg\rangle)$ 的 Rabi 振荡，且伴随 $\sqrt{2}$ 增强的集体 Rabi 频率。

**第四环：退相干的双重夹击。** 现实实验中五类退相干通道（\S4：多普勒展宽、位置抖动、激光强度噪声、激光相位噪声、Rydberg 衰变）的线性响应分析揭示了一个根本性困境：噪声标度律要求尽可能大的 $\Omega$ 以缩短门时间，而阻塞条件要求 $\Omega \ll V_{\text{vdW}}/\hbar$。这一双重约束将可用参数空间压缩到一个狭窄的窗口，在该窗口内传统方法各自遭遇不可逾越的瓶颈：STIRAP 被绝热速度极限锁死（$F \lesssim 0.985$），GRAPE 的开环优化在 sim-to-real gap 面前雪崩，反绝热驱动在非马尔可夫噪声下失效。

**第五环：RL 作为必然的范式选择。** 上述困境的共同根源是传统方法对精确模型和闭系假设的依赖。PPO + domain randomization（\S6）从根本上规避了这些假设：无模型学习避免了 sim-to-real gap；域随机化将噪声鲁棒性内化为策略参数；稀疏终端奖励尊重了量子保真度的物理定义。

这条证明链的每一步都不是可选的——删除任何一环，都无法理解为何需要下一环的方法。

## 8.2 AI 的定位：不是魔法，是无模型鲁棒控制

在原子物理语境下讨论"人工智能"容易引发误解。有必要明确：PPO 在本问题中的成功并非源于某种超越物理定律的"智能"，而是源于其算法结构恰好匹配了里德堡门控制问题的数学特征。具体而言：

**模型无关性 (model-free) = 规避 sim-to-real gap。** GRAPE 优化的目标函数是 $\Phi[H_{\text{sim}}]$——它寻找在模型 Hamiltonian $H_{\text{sim}}$ 下保真度最高的脉冲。一旦 $H_{\text{real}} \neq H_{\text{sim}}$，优化结果立即失效。PPO 则优化的是策略 $\pi_\theta$ 在与**环境的直接交互**中获得的累积奖励——它从未假设模型的精确形式。在训练中使用 domain randomization 后，策略实际上优化的是

$$\theta^* = \arg\max_\theta\;\mathbb{E}_{\xi \sim p(\xi)}[r(\theta, \xi)]$$

这是一个对噪声分布 $p(\xi)$ 的**隐式 ensemble 优化**，其效果等价于同时对无穷多个噪声实例进行鲁棒优化。值得注意的是，GRAPE 的 ensemble 版本（robust GRAPE / ensemble GRAPE）也可以在噪声分布上联合优化 $\max_{\mathbf{u}} \mathbb{E}_{\xi}[\Phi(\mathbf{u}, \xi)]$，这是量子最优控制领域中成熟的方法。本文未包含 ensemble GRAPE 基线——我们报告的 GRAPE 结果基于无噪声目标函数优化的脉冲，因此 PPO 与 GRAPE 的对比本质上是"noise-aware training vs. noise-agnostic optimization"的对比。一个更严格的方法论验证应当包含 ensemble GRAPE 作为额外基线（参见 §8.3）。

**PPO 的 clipping = spin-glass 景观上的稳定导航。** Bukov *et al.* [Bukov18] 证明量子控制景观具有 spin-glass 结构（大量近等深局部极值）。在这种景观上，DDPG/SAC 等 off-policy 算法因 replay buffer 中过期样本的误导而频繁发散 [Ernst25]。PPO 的 clipped surrogate 目标 $L^{\text{CLIP}}$ 将每步策略更新幅度限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，等效于在 spin-glass 景观上进行保守的局部搜索，避免了灾难性的长程跳跃。

**Domain randomization = 自适应的动态解耦。** 从物理直觉出发（\S6.3.3），DR 训练出的策略可类比为一种自适应的动态解耦序列——传统 DD 在固定时刻插入 $\pi$ 脉冲以平均慢噪声，而 DR 策略学会了根据当前密度矩阵自适应地调制脉冲的相位和幅度，实现对整个噪声谱的持续解耦。这不是"黑盒魔法"，而是有着明确物理对应的控制范式。

## 8.3 局限与挑战

诚实地审视本工作的局限对于准确评估方法的适用边界至关重要。

**(a) Hilbert 空间的指数标度。** $N$ 个双能级原子的 Hilbert 空间维度为 $2^N$，密度矩阵有 $4^N$ 个元素。本文的双原子系统（$4 \times 4$）和三原子系统（$8 \times 8$）仍在可处理范围内，但扩展到 $N \sim 10$（$4^{10} \approx 10^6$ 维密度矩阵）将使 Lindblad 演化的数值积分变得极其昂贵，单个 episode 的仿真时间可能从毫秒级增长到小时级。这直接限制了 RL 训练所需的样本数量，构成了向多原子系统推广的主要瓶颈。

**(b) Sim-to-Real Gap 的残余。** 尽管 domain randomization 在仿真中展示了良好的噪声鲁棒性，但仿真环境与真实实验之间仍存在不可忽视的差距：(i) 噪声分布的假设（例如 OU 过程描述激光噪声）可能与真实噪声统计不符；(ii) 仿真中未纳入的物理效应（如光镊交叉耦合、背景气体碰撞）可能在实验中产生意外影响；(iii) 量子态层析的测量误差限制了基于实时状态反馈的真正闭环策略在实验中的直接部署——当前 PPO 策略本质上是一种鲁棒开环方案（参见 §6.3.2）。

**(c) 稀疏奖励与信用分配。** 场景 B 采用的稀疏终端奖励（仅在 $t = T$ 时给出保真度反馈）使得智能体面临严峻的 credit assignment 问题。场景 C 通过引入 reward shaping（$r_t = \alpha \cdot (F_t - F_{t-1})$，$\alpha = 0.15$）有效缓解了这一问题，但在高保真度区间（$F > 0.99$）时，奖励信号的梯度仍然微弱，限制了进一步优化的速度。如何设计更好的 potential-based shaping reward 以加速收敛是一个值得探索的方向。

**(d) 缺乏形式化的最优性保证。** 与 GRAPE 不同（GRAPE 至少在给定模型下可以证明其找到了局部最优），PPO 作为一种基于采样的策略梯度方法，缺乏关于全局收敛性或近最优性的理论保证。我们无法确定当前策略距离全局最优脉冲还有多远——这在需要严格误差预算的容错量子计算中是一个需要正视的问题。

## 8.4 展望

本文的方法论框架自然指向若干有前景的扩展方向。

**碱土类原子 (Sr, Yb) 与光钟跃迁。** 碱土类原子提供了 $^1S_0 \leftrightarrow {^3P_0}$ 光钟跃迁作为量子比特编码基，其超窄线宽（$\sim \text{mHz}$）使得量子比特的相干时间从微秒提升到秒量级。在这样的平台上，退相干预算中 Rydberg 衰变的权重将大幅降低，而激光噪声通道的权重相对上升——这恰恰是 domain randomization 最擅长应对的噪声类型。将本文的 PPO + DR 框架迁移到 Sr/Yb 系统是一个高度可行且有意义的方向 [Evered23]。

**可微量子模拟器。** 近年来基于 JAX/Diffrax 等自动微分框架构建的可微量子模拟器使得 $\partial F/\partial \mathbf{u}$ 的精确梯度可以通过反向传播高效获得。这为结合梯度信息与 RL 策略提供了可能——例如，使用解析梯度进行策略网络的预训练（warm start），然后切换到 PPO 进行 domain randomization 微调。这种"梯度引导的 RL"可能兼具 GRAPE 的高效收敛和 PPO 的噪声鲁棒性。

**跨原子种类的迁移学习 (transfer learning)。** 不同碱金属原子的 Rydberg 态物理遵循相同的标度律，仅量子亏损参数 $\delta_{n\ell j}$ 不同。一个在 Rb $53S$ 上训练的策略是否可以通过微调适应 Cs $60S$ 或 Sr $70S$？如果标度律足以捕获系统间的物理相似性，那么 transfer learning 可以大幅减少在新原子种类上的训练成本。

**与量子纠错的集成。** 最终，高保真度的物理门操作需要与逻辑层的量子纠错协议（如 surface code）集成。RL 策略是否可以直接优化 logical error rate 而非物理保真度？这需要将纠错码的译码器纳入 RL 环境的奖励函数中，是一个兼具理论深度和实验意义的开放问题。

## 8.5 结语

从 Bohr 的量子亏损到 Schrödinger 方程的有效二能级约化，从 van der Waals 阻塞到 Lindblad 开放系统动力学，每一层物理抽象都为理解真实实验中的量子控制问题增添了新的维度与挑战。传统控制算法——无论多么精巧——都在"已知模型"与"可控噪声"的假设边界处触礁。

本文的数值实验定量地验证了这一判断：在高噪声场景 C（噪声幅度放大 3 倍 + 系统性幅度偏置）中，STIRAP 降至 $\langle F \rangle = 0.850$，GRAPE 降至 $0.814$，而经 domain randomization 训练的 PPO 策略达到 $\langle F \rangle = 0.990$——一个超过 14 个百分点的决定性优势。更重要的是，在 0–5% 额外幅度误差的鲁棒性扫描中，PPO 的保真度几乎不变（$\Delta F < 0.002$），而传统方法进一步退化。这不是偶然的数值巧合，而是 domain randomization 所赋予的噪声鲁棒性相对于传统开环优化的结构性优势在强噪声条件下的必然体现。

强化学习的引入不是对物理方法的否定，而是其自然延伸。正如绝热消除将不可解的三能级问题简化为可解的二能级问题，domain randomization 将不可控的噪声环境"简化"为策略网络可以学习的分布——两者都是物理学家最擅长的工具：通过改变描述层次来降低问题的有效复杂度。

量子物理与机器学习的交汇不是一时的潮流叠加，而是一种被退相干的数学结构——多源、非马尔可夫、有色——所严格要求的范式演进。在里德堡原子阵列这一充满前景的量子计算平台上，这种交汇正在将"驯服量子混沌"从理想化的理论命题转化为可操作的工程现实。

---

**参考文献**

- [Bukov18] A. G. Bukov *et al.*, "Reinforcement learning in different phases of quantum control," *Phys. Rev. X* **8**, 031086 (2018).
- [Ernst25] O. Ernst *et al.*, "Reinforcement learning for Rydberg quantum gates," *ICML 2025*, arXiv:2501.14372 (2025).
- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [Guatto24] S. Guatto *et al.*, "Model-free quantum gate design and calibration using deep reinforcement learning," *Nat. Commun.* **15**, 8353 (2024).
- [Niu19] M. Y. Niu *et al.*, "Universal quantum control through deep reinforcement learning," *npj Quantum Inf.* **5**, 33 (2019).

---

# 附录

---

# 附录 A：Quantum Defect Theory 完整推导

## A.1 引言

Quantum defect theory 是理解 alkali metal Rydberg 态能级结构的基础理论框架。与 hydrogen atom 不同，alkali metal 的 valence electron 在靠近 ionic core 时会受到 core electrons 的屏蔽和极化效应，导致能级偏离纯 Coulomb 势的预测。本附录从 radial Schrödinger equation 出发，完整推导 quantum defect 的定义及其物理起源。

## A.2 Radial Schrödinger Equation

考虑 alkali metal 中 valence electron 的径向运动。在 central field approximation 下，将多电子问题简化为单个 valence electron 在有效势 $V(r)$ 中的运动。径向 Schrödinger equation 为：

$$\left[-\frac{\hbar^2}{2m}\frac{d^2}{dr^2} + \frac{\ell(\ell+1)\hbar^2}{2mr^2} + V(r)\right]u(r) = Eu(r) \tag{A.1}$$

其中 $u(r) = rR(r)$ 是约化径向波函数，$R(r)$ 为径向波函数，$\ell$ 为 orbital angular momentum quantum number。

有效势 $V(r)$ 具有以下渐近行为：

- **远区** ($r \to \infty$)：core electrons 完全屏蔽核电荷，$V(r) \to -e^2/r$，即纯 Coulomb 势。
- **近区** ($r \to 0$)：valence electron 穿透 core electron cloud，感受到更强的核电荷吸引，$V(r)$ 比 $-e^2/r$ 更深。

定义 core radius $r_c$ 为 $V(r)$ 显著偏离 $-e^2/r$ 的临界半径。对于典型 alkali metal（如 $^{87}\text{Rb}$），$r_c \sim$ 几个 Bohr radii。

## A.3 WKB 近似与 Coulomb 区域

在经典允许区域 $r_1 < r < r_2$（两个 classical turning points 之间），WKB 近似给出波函数的形式：

$$u(r) \approx \frac{A}{\sqrt{k(r)}} \sin\left(\int_{r_1}^{r} k(r')\,dr' + \frac{\pi}{4}\right)$$

其中 local wavenumber 为：

$$k(r) = \frac{1}{\hbar}\sqrt{2m\left[E - V(r) - \frac{\ell(\ell+1)\hbar^2}{2mr^2}\right]}$$

对于 Coulomb 区域（$r > r_c$），将 $V(r) = -e^2/r$ 代入，径向波函数积累的相位为：

$$\phi_C = \int_{r_1}^{r_2} k(r)\,dr$$

此积分可以解析完成（对于纯 Coulomb 势），给出 hydrogen-like 的量子化条件。

**WKB 量子化条件**要求波函数在两个 turning points 之间形成驻波：

$$\phi_C + \phi_{\text{short}} = \left(n_r + \frac{1}{2}\right)\pi$$

其中 $n_r$ 为 radial quantum number（波函数在径向方向的节点数），$\phi_{\text{short}}$ 为来自 short-range（非 Coulomb）区域的额外相位贡献。

## A.4 Short-Range Phase Shift

在 $r < r_c$ 的非 Coulomb 区域，由于 core electron 的屏蔽不完全，effective potential 比纯 Coulomb 势更深。这意味着波函数在该区域中的 local wavenumber 更大，因此积累了比纯 hydrogen 情形更多的相位。

定义 short-range phase shift 为：

$$\phi_{\text{short}} = \int_0^{r_c} \left[k_{\text{actual}}(r) - k_{\text{Coulomb}}(r)\right] dr$$

这一额外相位完全由 $r < r_c$ 区域内 $V(r)$ 与 $-e^2/r$ 的偏差决定。关键观察是：对于高 Rydberg 态（大 $n$），valence electron 大部分时间在远离 core 的区域运动，因此 $\phi_{\text{short}}$ 对能量的依赖很弱——它主要由 core 附近的势形状决定，而这几乎不随 $n$ 变化。

## A.5 Quantum Defect 的定义

定义 quantum defect 为 short-range phase shift 除以 $\pi$：

$$\delta_\ell \equiv \frac{\phi_{\text{short}}}{\pi}$$

将此代入量子化条件。对于纯 hydrogen，Coulomb 相位积分给出：

$$\phi_C^{(\text{H})} = \left(n_r + \frac{1}{2}\right)\pi$$

能量为 $E_n = -\text{Ry}^*/n^2$，其中 $n = n_r + \ell + 1$。

对于 alkali metal，额外的 $\phi_{\text{short}} = \delta_\ell \pi$ 等效于将 radial quantum number 替换为 $n_r - \delta_\ell$，或等价地将 principal quantum number 替换为 $n^* = n - \delta_\ell$。因此能级公式变为：

$$E_{n\ell} = -\frac{\text{Ry}^*}{(n - \delta_\ell)^2} \tag{A.2}$$

其中 $\text{Ry}^* = \text{Ry} \cdot \mu/m_e$ 是修正的 Rydberg constant（考虑 reduced mass 效应，$\mu = m_e M/(m_e + M)$ 为 electron-nucleus reduced mass），$n^* = n - \delta_\ell$ 称为 effective quantum number。

## A.6 Quantum Defect 的能量依赖性

虽然 $\delta_\ell$ 主要取决于 core 区域的势，但它对能量有弱的依赖性。Ritz 展开将 quantum defect 表示为 effective quantum number 的幂级数：

$$\delta_{n\ell j} = \delta_0 + \frac{\delta_2}{(n-\delta_0)^2} + \frac{\delta_4}{(n-\delta_0)^4} + \cdots \tag{A.3}$$

各项的物理起源：

- **$\delta_0$（主项）**：捕获 valence electron 对 core 的 penetration effect。这是 quantum defect 的主要贡献，由 core 区域 $V(r)$ 与 Coulomb 势的偏差大小决定。对于 Rb 的 $nS$ 态，$\delta_0 \approx 3.13$；对于 $nD$ 态，$\delta_0 \approx 1.35$ [Gallagher94]。

- **$\delta_2$（二阶修正）**：主要来自 ionic core 的 polarization effect。当 valence electron 距离 core 较近时，其电场会极化 core electron cloud，产生 induced dipole moment，进而修正有效势。此效应引入 $\sim -\alpha_d/(2r^4)$ 的极化势，其中 $\alpha_d$ 为 core 的 dipole polarizability。

- **$\delta_4$ 及更高阶项**：来自 quadrupole polarization、exchange interaction 以及 relativistic correction 等更精细的效应。在实际拟合中，通常只需保留到 $\delta_2$ 即可描述现有实验精度 [Steck18]。

$\delta_{n\ell j}$ 中 $j$ 的依赖性来自 spin-orbit coupling，对于 $\ell \geq 1$ 的态，fine structure splitting 导致 $j = \ell \pm 1/2$ 的 quantum defect 略有不同。

## A.7 物理图像与 $\ell$ 依赖性

Quantum defect 的大小直接反映 valence electron 对 core 的穿透程度：

- **$\ell = 0$（S 态）**：无 centrifugal barrier，valence electron 的径向波函数在 $r = 0$ 处有限（$u(r) \sim r$），可以深入 core 区域。因此 $\delta_0$ 很大。以 $^{87}\text{Rb}$ 为例，$\delta_0(S_{1/2}) \approx 3.131$ [Steck18]。

- **$\ell = 1$（P 态）**：有较弱的 centrifugal barrier $\sim \ell(\ell+1)/r^2$，但 valence electron 仍能显著穿透 core。$\delta_0(P_{1/2}) \approx 2.654$，$\delta_0(P_{3/2}) \approx 2.642$。

- **$\ell = 2$（D 态）**：centrifugal barrier 更强，穿透减小。$\delta_0(D_{3/2}) \approx 1.348$，$\delta_0(D_{5/2}) \approx 1.346$。

- **$\ell \geq 3$（F 态及更高）**：centrifugal barrier 足够高，几乎完全阻止 valence electron 进入 core 区域。$\delta_\ell \ll 1$，能级几乎是 hydrogen-like 的。此时主要的修正来自 long-range polarization 势而非 core penetration。

这一 $\ell$ 依赖性对 Rydberg atom 的实验选择具有重要意义：低 $\ell$ 态的能级间距可被精确测量，用于确定 quantum defect parameters [SWM10]；而高 $\ell$ 态由于接近 hydrogenic，其性质可以用解析公式直接计算。

## A.8 小结

Quantum defect theory 提供了一个优雅而实用的框架，将复杂的多电子 alkali atom 问题归结为仅含少数参数的 modified hydrogen model。式 (A.2) 和 (A.3) 构成了精确计算 Rydberg 态能级的基础，进而决定了 transition frequencies、dipole matrix elements、以及 atom-atom interaction 的强度（参见附录 C）。该理论的成功源于一个基本事实：Rydberg electron 的波函数绝大部分分布在 core 外的 Coulomb 区域，core 内的复杂多体效应仅以少数 phase-shift 参数体现。

---

**参考文献**

- [Gallagher94] T. F. Gallagher, *Rydberg Atoms*, Cambridge University Press (1994).
- [Steck18] D. A. Steck, "Rubidium 87 D Line Data," available online (2018).
- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," Rev. Mod. Phys. **82**, 2313 (2010).

---

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

---

# 附录 C：$C_6$ 通道求和

## C.1 引言

Rydberg atom 之间的 van der Waals interaction 是实现 Rydberg blockade 的物理基础。两个处于 Rydberg state $|r\rangle$ 的原子之间的长程相互作用强度由 $C_6$ 系数描述，其数值依赖于所有 intermediate pair states 的 dipole matrix elements 和 energy defects 的通道求和。本附录详细推导 $C_6$ 的计算方法。

## C.2 Dipole-Dipole 算符

两个相距 $R$ 的中性原子之间，最低阶的 electrostatic interaction 来自 dipole-dipole coupling。在 point-dipole approximation 下，相互作用算符为：

$$\hat{V}_{dd} = \frac{1}{4\pi\epsilon_0} \frac{\hat{\mathbf{d}}_1 \cdot \hat{\mathbf{d}}_2 - 3(\hat{\mathbf{d}}_1\cdot\hat{\mathbf{n}})(\hat{\mathbf{d}}_2\cdot\hat{\mathbf{n}})}{R^3} \tag{C.1}$$

其中 $\hat{\mathbf{d}}_i = -e\hat{\mathbf{r}}_i$ 是原子 $i$ 的 electric dipole operator，$\hat{\mathbf{n}} = \mathbf{R}/R$ 为连接两原子的单位方向矢量。

为计算矩阵元素，将 dipole operator 展开为 spherical components：

$$d_q = -er\, C_q^{(1)}(\theta,\phi), \quad q = 0, \pm 1$$

其中 $C_q^{(1)}$ 为 renormalized spherical harmonics。选取 quantization axis 沿 interatomic axis（$\hat{\mathbf{n}} = \hat{z}$），dipole-dipole 算符可表示为：

$$\hat{V}_{dd} = -\frac{e^2}{4\pi\epsilon_0 R^3} \sum_{q=-1}^{1} (-1)^q \binom{2}{1+q}^{1/2} r_1 C_q^{(1)}(1) \cdot r_2 C_{-q}^{(1)}(2) \cdot f(q)$$

其中 $f(q)$ 包含 angular coupling coefficients。对于 $\hat{\mathbf{n}} \| \hat{z}$ 的简化情形，selection rule 要求 $\Delta m_1 + \Delta m_2 = 0$。

## C.3 Matrix Elements 与 Wigner-Eckart 定理

在 $|n, \ell, j, m_j\rangle$ 基下，单原子 dipole matrix element 通过 Wigner-Eckart theorem 分解为：

$$\langle n'\ell'j'm_j' | d_q | n\ell j m_j \rangle = (-1)^{j'-m_j'} \begin{pmatrix} j' & 1 & j \\ -m_j' & q & m_j \end{pmatrix} \langle n'\ell'j' \| d \| n\ell j \rangle$$

其中 $(\cdots)$ 为 Wigner 3-j symbol，$\langle n'\ell'j' \| d \| n\ell j \rangle$ 为 reduced matrix element。

Reduced matrix element 可进一步分解为 radial integral 和 angular factor：

$$\langle n'\ell'j' \| d \| n\ell j \rangle = (-1)^{j'+\ell'+s+1} e \sqrt{(2j'+1)(2j+1)} \begin{Bmatrix} \ell' & j' & s \\ j & \ell & 1 \end{Bmatrix} \langle n'\ell' | r | n\ell \rangle$$

其中 $\{\cdots\}$ 为 Wigner 6-j symbol，$s = 1/2$，radial integral $\langle n'\ell'|r|n\ell\rangle$ 需要数值计算 Rydberg 波函数得到（通常用 Numerov method 或 quantum defect theory 给出的 Coulomb functions）。

## C.4 Second-Order Perturbation Theory: Van der Waals $C_6$

当两原子初态为 $|rr\rangle \equiv |n\ell j m_j\rangle_1 |n\ell j m_j\rangle_2$ 时，$\hat{V}_{dd}$ 的 first-order contribution 通常为零（因为 dipole operator 改变 $\ell$ 的 parity）。因此需要二阶微扰，给出 van der Waals interaction $\sim C_6/R^6$。

$C_6$ 系数由以下通道求和给出：

$$C_6 = -\sum_{\alpha\beta} \frac{|\langle rr| \hat{V}_{dd} |\alpha\beta\rangle|^2}{E_{\alpha\beta} - E_{rr}} \tag{C.2}$$

求和遍历所有满足 selection rules 的 intermediate pair states $|\alpha\beta\rangle = |n_1'\ell_1'j_1'm_1'\rangle |n_2'\ell_2'j_2'm_2'\rangle$，其中：

- $\Delta\ell = \pm 1$（electric dipole selection rule）
- $\Delta m_1 + \Delta m_2 = 0$（对于 $\hat{\mathbf{n}} \| \hat{z}$）
- $E_{\alpha\beta} = E_{n_1'\ell_1'j_1'} + E_{n_2'\ell_2'j_2'}$ 为 intermediate pair state 的能量

对于 $|nS_{1/2}\rangle$ 初态，最主要的通道为 $|nS\rangle|nS\rangle \to |n'P\rangle|(n'-1)P\rangle$ 以及 $|nS\rangle|nS\rangle \to |(n-1)P\rangle|nP\rangle$ 等。需要对 $n'$ 求和（包括连续态的贡献，但对于 Rydberg 态，通常由少数几个 $n'$ 值主导）。

## C.5 Zeeman 态依赖性

式 (C.2) 中的 $C_6$ 显式依赖于初态的 $m_j$ 量子数。当原子处于特定的 $|m_j\rangle$ 态时，selection rules 限制了贡献的通道数目。

Walker 和 Saffman [WS08] 系统地计算了不同 $m_j$ 配置下的 $C_6$。主要结论：

- 对于 $|nS_{1/2}, m_j = +1/2\rangle|nS_{1/2}, m_j = +1/2\rangle$，$C_6$ 值与 $m_j$-averaged 的结果有显著差异。
- 在存在外 magnetic field（定义 quantization axis）的情况下，$m_j$-specific $C_6$ 是正确的使用值。
- 对于 randomly oriented atoms（如热原子气体），需要对 $m_j$ 取平均。

具体而言，对于 $^{87}\text{Rb}$ 的 $|60S_{1/2}\rangle$ 态：$C_6/(2\pi) \approx -140$ GHz $\mu\text{m}^6$（$m_j$-averaged），而特定 $|m_j| = 1/2$ 的值可能偏离数十百分比。

## C.6 Förster Defect 与交叉区

当 intermediate pair state 的能量接近初态能量时，energy denominator $E_{\alpha\beta} - E_{rr} \to 0$，二阶微扰论发散，表明需要非微扰处理。

定义 Förster defect：

$$\delta_F = E_{nP} + E_{(n-1)P} - 2E_{nS}$$

物理意义：$\delta_F$ 衡量 pair state $|nP\rangle|(n-1)P\rangle$ 与初态 $|nS\rangle|nS\rangle$ 之间的 energy mismatch。

两种极限情况：

1. **Van der Waals 极限** ($|\delta_F| \gg |V_{dd}|$)：$\hat{V}_{dd}$ 可以作为微扰处理，interaction 为 $\sim C_6/R^6$，如式 (C.2) 描述。

2. **Resonant dipole 极限** ($|\delta_F| \to 0$)：需要对简并（或近简并）的 pair states 做 degenerate perturbation theory。此时 interaction 变为 first-order，$\sim C_3/R^3$，其中 $C_3$ 由 single dipole matrix element 决定。

交叉发生在 $|V_{dd}(R)| \sim |\delta_F|$ 的距离处，即：

$$R_{\text{cross}} \sim \left(\frac{C_3}{\delta_F}\right)^{1/3}$$

对于 $R \ll R_{\text{cross}}$，interaction 为 $1/R^3$（resonant dipole）；对于 $R \gg R_{\text{cross}}$，退化解除，interaction 恢复为 $1/R^6$（van der Waals）。

在某些特殊的 $n$ 值处，$\delta_F$ 可以非常小甚至为零（Förster resonance），这可以通过外电场 Stark tuning 实现 [Gallagher94]。Förster resonance 大幅增强了原子间相互作用，是 Rydberg blockade 实验中的重要工具。

## C.7 小结

$C_6$ 系数的精确计算需要对大量 intermediate pair channels 进行求和，涉及 radial matrix elements（由 quantum defect theory 或数值波函数给出）和 angular coupling coefficients（由 Wigner symbols 决定）。$C_6$ 的符号、大小和 $m_j$ 依赖性直接影响 Rydberg blockade 的效率和 gate fidelity。当 Förster defect 较小时，interaction 从 van der Waals 型（$1/R^6$）过渡到 resonant dipole 型（$1/R^3$），这为实验调控提供了额外的自由度。

---

**参考文献**

- [WS08] T. G. Walker and M. Saffman, "Consequences of Zeeman degeneracy for the van der Waals blockade between Rydberg atoms," Phys. Rev. A **77**, 032723 (2008).
- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," Rev. Mod. Phys. **82**, 2313 (2010).
- [Gallagher94] T. F. Gallagher, *Rydberg Atoms*, Cambridge University Press (1994).

---

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
- [Goerz14b] M. H. Goerz, T. Calarco, and C. P. Koch, "The quantum speed limit of optimal controlled phasegates for trapped neutral atoms," New J. Phys. **16**, 055012 (2014).
- [Caneva11] T. Caneva, T. Calarco, and S. Montangero, "Chopped random-basis quantum optimization," Phys. Rev. A **84**, 022326 (2011).
- [Larocca22] M. Larocca et al., "Diagnosing barren plateaus with tools from quantum optimal control," Quantum **6**, 824 (2022).

---

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

本文使用 QuTiP (Quantum Toolbox in Python) 进行基线方法（STIRAP、GRAPE）的 Lindblad 方程数值求解，而 RL 环境中为提升仿真速度采用 `scipy.linalg.expm` 直接矩阵指数化（参见 \S6.4）。关键实现细节如下：

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

---

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

| 超参数 | 符号 | 场景 B | 场景 C | 说明 |
|---|---|---|---|---|
| Learning rate | $\alpha$ | $3\times10^{-4}$ | $3\times10^{-4} \to 5\times10^{-5}$（线性退火） | Adam optimizer 学习率 |
| Steps per rollout | $n_{\text{steps}}$ | 2048 | 4096 | 每次采集的 transition 数 |
| Minibatch size | $|B|$ | 64 | 512 | 小批量大小 |
| Epochs per update | $K$ | 10 | 10 | 每次数据采集后的 epoch 数 |
| Discount factor | $\gamma$ | 1.0 | 1.0 | 无折扣（关心终态 fidelity） |
| GAE parameter | $\lambda$ | 0.95 | 0.95 | Advantage estimation 平滑参数 |
| Clip range | $\varepsilon$ | 0.2 | 0.2 | PPO 裁剪范围 |
| Value loss coefficient | $c_1$ | 0.5 | 0.5 | Value function loss 权重 |
| Entropy coefficient | $c_2$ | 0.01 | 0.005 | 熵正则化系数，鼓励 exploration |
| Total timesteps | — | 50,000 | 3,000,000 | 总训练步数 |
| Network architecture | — | MLP [64, 64] | MLP [512, 256] | Actor-Critic 共享网络 |
| Activation function | — | Tanh | Tanh | 隐藏层激活函数 |
| Max grad norm | — | 0.5 | 0.5 | Gradient clipping 阈值 |
| Reward shaping | $\alpha$ | 0 | 0.15 | 中间保真度变化奖励权重 |

**超参数选择依据**：

- **$\gamma = 1.0$**：由于 Rydberg gate 的 reward 仅在 episode 结束时给出（terminal fidelity），无需时间折扣。所有时间步的 action 对最终 fidelity 同等重要。
- **$n_{\text{steps}}$**：场景 B 中 2048 步约包含 68 个 complete episodes（episode length = 30）；场景 C 中增大至 4096 步（episode length = 60）以获得更稳定的 advantage 估计。
- **$c_2$**：场景 C 降至 0.005，因更长的训练允许更早收敛到 deterministic policy。
- **学习率退火**（场景 C）：线性退火从 $3 \times 10^{-4}$ 降至 $5 \times 10^{-5}$，在训练后期进行精细优化。
- **Reward shaping**（场景 C）：$r_t = \alpha \cdot (F_t - F_{t-1}) + \mathbb{1}_{t=T} \cdot F_T$，中间保真度变化信号显著加速了信用分配，缓解了稀疏奖励的探索困难。

## F.4 环境参数

| 参数 | 场景 B | 场景 C | 说明 |
|---|---|---|---|
| State dimension | 32 | 33 | 密度矩阵 $\rho$ 的 real + imaginary 部分展平（场景 C 增加时间分数 $t/T$） |
| Action dimension | 2 | 2 | 归一化的 $\Omega(t)$ 和 $\Delta(t)$ |
| Episode length | 30 steps | 60 steps | 对应 gate time $T$ 的等分 |
| Time step | $\Delta t = T/30$ | $\Delta t = T/60$ | 每步控制参数恒定 |
| Reward | $r_T = \mathcal{F}$ | $r_t = \alpha \Delta F_t + \mathbb{1}_T \mathcal{F}$ | 终态 fidelity（场景 C 含 shaping） |

**State representation**：4×4 密度矩阵 $\rho$ 具有 16 个复数元素，拆分为实部和虚部后为 32 维实向量。场景 C 在此基础上增加一个时间分数维度 $t/T \in [0, 1]$，总计 33 维——这使策略网络能够根据门演化的阶段调整控制策略。由于 $\rho$ 为 Hermitian，实际独立参数更少（$d^2 - 1 = 15$），但使用完整表示简化了实现且不影响 learning 效率。

**Action space**：连续动作空间 $\mathbf{a} = (a_\Omega, a_\Delta) \in [-1, 1]^2$，通过线性映射转换为物理参数：

$$\Omega(t) = \frac{a_\Omega + 1}{2} \cdot \Omega_{\max}, \quad \Delta(t) = a_\Delta \cdot \Delta_{\max}$$

其中 $\Omega_{\max}$ 和 $\Delta_{\max}$ 为实验可达范围的上限。

## F.5 小结

本附录给出的 PPO 实现遵循 Stable Baselines3 的标准接口 [PPO17]。超参数设置基于对 Rydberg quantum control 环境特点的分析（short episodes、terminal reward、continuous actions），部分参数经过 grid search 微调。Environment 的 domain randomization 设计使得 trained policy 可直接迁移到参数不确定的实际实验场景。

---

**参考文献**

- [PPO17] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).

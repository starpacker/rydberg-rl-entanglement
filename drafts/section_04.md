# 4 退相干预算与传统控制算法的失效

在 \S1–\S3 中，我们建立了理想化的 Rydberg blockade 物理图景：完美的阻塞条件下，一个精确的 $\pi$ 脉冲即可将 $|gg\rangle$ 确定性地转化为最大纠缠态 $|W\rangle$。然而，现实实验远非如此简洁——多种退相干源同时侵蚀门保真度，而传统控制算法各自面临根本性的局限。本节将系统性地建立退相干预算（decoherence budget），推导保真度对噪声的线性响应理论，并逐一分析三类经典控制算法的失效模式，从而为引入无模型强化学习方法提供严格的物理动机。

## 4.1 退相干来源的全景图

真实实验中的退相干通道远不止自发辐射一项。表 2 汇总了 $^{87}\text{Rb}$ $53S_{1/2}$ 态在 $T = 10\;\mu\text{K}$ 典型实验条件下的所有主要退相干来源。

**表 2**：退相干来源全景（Rb $53S$, $T = 10\;\mu\text{K}$, $R = 2\;\mu\text{m}$）

| 来源 | 数学描述 | 典型值 | 引用 |
|------|----------|--------|------|
| 自发辐射 + BBR | Lindblad $L_k = \sqrt{\Gamma_r}\|g\rangle\langle r\|$ | $\tau_{\text{eff}} \approx 88\;\mu\text{s}$ | [Beterov09] |
| 中间态散射 (5P/6P) | $R_{\text{sc}} = \gamma_{5P}\Omega_1^2/(4\Delta^2)$ | $\sim 10^{-4}/\mu\text{s}$ | [Evered23] |
| 多普勒位移 | $\delta\omega = \mathbf{k}_{\text{eff}}\cdot\mathbf{v}$, $\sigma_v = \sqrt{k_BT/m}$ | $\sigma_\omega/2\pi \approx 50\;\text{kHz}$ | [dL18] |
| 位置抖动 ($1/R^6$ 灵敏) | $\delta V = -6V_{\text{vdW}}\,\delta R/R$ | $\sigma_R \approx 100\;\text{nm}$ | [dL18] |
| 激光强度 OU 噪声 | $\Omega(t) = \Omega_0(1+\xi(t))$, OU process | $\delta\Omega/\Omega \sim 1\text{--}2\%$ | [Day22] |
| 激光相位噪声 (servo bump) | PSD $S_\phi(f)$ 尖峰 at $f \sim \Omega$ | $-80\;\text{dBc/Hz}$ | [PRXQ25] |

以下逐项阐述各退相干通道的物理起源及其对门操作的影响。

**自发辐射与黑体辐射 (BBR)**。如 \S1.3 所述，Rydberg 态通过自发辐射和 BBR 诱导跃迁衰变回低能级。在 Lindblad 开放系统框架中，该过程由塌缩算符 $L_k = \sqrt{\Gamma_r}|g\rangle\langle r|$ 描述，其中 $\Gamma_r = 1/\tau_{\text{eff}}$。对于 Rb $53S$ 态，$\tau_{\text{eff}} \approx 88\;\mu\text{s}$ [Beterov09]，看似远长于典型门时间（$\sim 0.3\text{--}1\;\mu\text{s}$），但由于门操作要求的保真度极高（$F > 0.999$），即使是 $\sim 10^{-3}$ 量级的衰变概率也构成显著的误差贡献。

**中间态散射**。双光子激发方案（\S2.2）中，中间态 $5P_{3/2}$（或 $6P_{3/2}$）虽被大失谐 $\Delta$ 抑制，但仍有非零的布居概率 $\sim \Omega_1^2/(4\Delta^2)$。中间态的自发辐射速率 $\gamma_{5P} \approx 6\;\text{MHz}$ 导致不可逆的散射事件，每次散射使量子相干性完全丧失。散射速率 $R_{\text{sc}} = \gamma_{5P}\Omega_1^2/(4\Delta^2)$ 在 Evered 等人 [Evered23] 的实验中约为 $\sim 10^{-4}/\mu\text{s}$——虽然绝对数值很小，但在多次门操作的累积下不可忽视。

**多普勒位移**。尽管原子被冷却到 $T \sim 10\;\mu\text{K}$，残余热运动仍产生多普勒频移 $\delta\omega = \mathbf{k}_{\text{eff}} \cdot \mathbf{v}$，其中 $\mathbf{k}_{\text{eff}} = \mathbf{k}_1 + \mathbf{k}_2$ 为双光子有效波矢。对于反向传播的 420 nm + 1013 nm 激光对，$k_{\text{eff}} \approx 2\pi \times 2.4\;\mu\text{m}^{-1}$，热速度 $\sigma_v = \sqrt{k_BT/m} \approx 0.03\;\text{m/s}$，给出多普勒展宽 $\sigma_\omega/2\pi \approx 50\;\text{kHz}$ [dL18]。每次实验 shot 中各原子的多普勒频移随机采样自 Gaussian 分布，构成一种 shot-to-shot 的准静态噪声。

**位置抖动与 $1/R^6$ 非线性放大**。光镊中原子的位置不确定度 $\sigma_R \approx 100\;\text{nm}$（由零点运动和热运动共同决定）通过 van der Waals 相互作用 $V = C_6/R^6$ 的极端非线性被放大为相互作用能的涨落：$\delta V/V = -6\,\delta R/R$ [dL18]。对于 $R = 2\;\mu\text{m}$ 的原子间距，$\sigma_R/R = 5\%$ 意味着 $\delta V/V \sim 30\%$——这是一个惊人的信号：微小的位置不确定度经过六次方放大后，成为阻塞条件精确度的主要威胁。

**激光强度 Ornstein-Uhlenbeck (OU) 噪声**。激光功率经 AOM/EOM 稳定后仍存在残余幅度抖动。将瞬时 Rabi 频率写为 $\Omega(t) = \Omega_0[1 + \xi(t)]$，其中 $\xi(t)$ 为零均值的 OU 过程（$\langle \xi(t)\xi(t')\rangle = (\sigma^2/2\theta)\exp(-\theta|t-t'|)$，关联时间 $\theta^{-1} \sim 10\;\mu\text{s}$），典型相对幅度噪声为 $\delta\Omega/\Omega \sim 1\text{--}2\%$ [Day22]。该噪声直接调制驱动强度，在脉冲面积敏感的操作中引入系统性误差。

**激光相位噪声与 servo bump**。激光器锁频伺服环路在带宽边缘（$f \sim \text{MHz}$）不可避免地产生相位噪声增强——即所谓的 servo bump。其功率谱密度 $S_\phi(f)$ 在 $f \approx \Omega/2\pi$ 附近出现尖峰，典型幅度约 $-80\;\text{dBc/Hz}$ [PRXQ25]。当 servo bump 频率与 Rabi 频率共振时，相位噪声被共振放大为退相干——这是一种特别隐蔽的噪声通道，因为它恰好在控制参数的特征频率处起作用。

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

其中 $T_{\text{QSL}} \sim 10/\Omega_{\text{eff}}$ 为绝热性所需的最短时间（quantum speed limit），$\tau_{\text{Ryd}} \approx 88\;\mu\text{s}$ 为 Rydberg 态有效寿命。为使 $T$ 足够短以抑制退相干，$\Omega_{\text{eff}}$ 必须足够大，但增大 $\Omega$ 又受阻于 blockade 条件。

定量地，在 STIRAP 兼容区域（$\Omega/2\pi < 1\;\text{MHz}$，$T \sim 5\;\mu\text{s}$），由式 (4.3)–(4.4) 可估算：多普勒贡献的保真度误差 $> 2 \times 10^{-3}$，Rydberg 衰变贡献 $> 10^{-2}$。两者叠加后，STIRAP 方案的保真度被天花板锁定在 $F \lesssim 0.985$ [Vitanov17, YB23]——远低于容错量子计算的阈值。

### (b) GRAPE (Gradient Ascent Pulse Engineering)

GRAPE 是由 Khaneja 等人 [Khaneja05] 提出的最优控制算法，通过将连续脉冲离散化为 $N$ 个时间片，在每个时间片上计算目标函数对控制参数的精确梯度，进行迭代优化。其梯度公式为

$$\frac{\partial \Phi}{\partial u_k(t_j)} = -i\Delta t\,\langle\psi_{\text{tgt}}| U_N\cdots [H_k, U_j]\cdots U_1|\psi_0\rangle + \text{c.c.} \tag{4.5}$$

其中 $\Phi$ 为保真度目标函数，$u_k(t_j)$ 为第 $j$ 个时间片上第 $k$ 个控制场的振幅，$U_j = \exp(-iH(t_j)\Delta t)$ 为相应的传播子。完整推导见附录 D。

GRAPE 及其变体（如 Krotov 算法 [Goerz14]）在闭系或弱噪声环境中可以找到极高保真度的脉冲方案。然而，在真实里德堡实验条件下，GRAPE 面临两个致命缺陷。

**致命缺陷 1：Sim-to-Real Gap（模型—现实鸿沟）**。GRAPE 的优化过程在模型 Hamiltonian $H_{\text{sim}}$ 上进行，找到的最优脉冲利用了 $H_{\text{sim}}$ 中精细的量子干涉路径来实现高保真度。然而，真实系统的 Hamiltonian 为 $H_{\text{real}} = H_{\text{sim}} + \delta H$，其中 $\delta H$ 包含了上述所有未精确建模的噪声通道。当 $\delta H$ 破坏了脉冲所依赖的干涉条件时，保真度不是渐进降低，而是发生**雪崩式崩溃**——Goerz 等人 [Goerz14] 的数值模拟明确显示，GRAPE/Krotov 脉冲在偏离设计参数窗口仅数个百分点后，保真度即灾难性下降。这一脆弱性的根源在于，GRAPE 本质上是一种**开环**（open-loop）优化：它优化的是单一确定模型下的性能，而非对参数不确定性的平均性能。

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
- [PRXQ25] "Sensitivity of quantum gate fidelity to laser phase and intensity noise," *PRX Quantum* **6**, 010331 (2025).
- [Vitanov17] N. V. Vitanov *et al.*, "Stimulated Raman adiabatic passage in physics, chemistry, and beyond," *Rev. Mod. Phys.* **89**, 015006 (2017).
- [YB23] P. Yagüe Bosch *et al.*, "Shortcuts to adiabaticity for Rydberg gate design," *Ann. Phys.* (2023). arXiv:2312.11594.
- [Khaneja05] N. Khaneja *et al.*, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent," *J. Magn. Reson.* **172**, 296 (2005).
- [Goerz14] M. H. Goerz *et al.*, "Optimal control theory for a unitary operation under dissipative evolution," *Phys. Rev. A* **90**, 032329 (2014).
- [Larocca22] M. Larocca *et al.*, "Diagnosing barren plateaus with tools from quantum optimal control," *Quantum* **6**, 824 (2022).
- [Berry09] M. V. Berry, "Transitionless quantum driving," *J. Phys. A* **42**, 365303 (2009).

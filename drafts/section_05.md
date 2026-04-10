# 5 实验设置：可量化的失效场景

本节定义精确且可复现的实验场景，定量地暴露传统控制方法的失效边界——为后续强化学习方案提供一致的基准测试平台。每一组参数均取自近期实验文献 [Evered23]，确保物理上的合理性与可验证性。

## 5.1 物理系统

我们选取 $^{87}$Rb 双原子系统作为核心平台。两个原子被分别囚禁在间距为 $R = 2\;\mu\text{m}$ 的光镊（optical tweezer）中。Rydberg 态选为

$$|r\rangle = |53S_{1/2},\; m_J = +\tfrac{1}{2}\rangle$$

其有效寿命（含 BBR 修正）为 $\tau_{\text{eff}} = 88\;\mu\text{s}$（参见 \S1.3 中黑体辐射对寿命的修正公式），满足门操作时长 $T_{\text{gate}} \ll \tau_{\text{eff}}$ 的基本要求。

**激发方案。** 采用经 $6P_{3/2}$ 中间态的双光子激发路径（参见 \S2.2），中间态失谐为

$$\Delta/2\pi = 7.8\;\text{GHz}$$

该失谐远大于 $6P_{3/2}$ 态的自然线宽（$\Gamma_{6P}/2\pi \approx 6\;\text{MHz}$），有效压制中间态的自发辐射（$\Gamma_{\text{eff}} \sim \Gamma_{6P} \cdot (\Omega_1/2\Delta)^2$），同时保持足够的有效 Rabi 频率。

**van der Waals 相互作用。** 在原子间距 $R = 2\;\mu\text{m}$ 下，由 $53S_{1/2}$ 态的 $C_6$ 系数计算 van der Waals 相互作用强度：

$$V_{\text{vdW}} = \frac{C_6}{R^6}, \qquad V_{\text{vdW}}/2\pi \approx 500\;\text{MHz} \tag{5.1}$$

相应的阻塞半径 $R_b = (C_6/\hbar\Omega)^{1/6} \gg R$，确保系统深处 blockade 极限。基线 Rabi 频率取为

$$\Omega/2\pi = 4.6\;\text{MHz}$$

与 Evered *et al.* 2023 [Evered23] 的实验参数一致。blockade 比 $V_{\text{vdW}}/\Omega \approx 109 \gg 1$，充分满足 \S3.2 中推导的阻塞条件。

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

其中 $\theta^{-1} = 10\;\mu\text{s}$ 为相关时间（与典型门操作时长同量级），$\sigma = 0.02$ 为归一化噪声强度，$dW_t$ 为标准 Wiener 增量。实际 Rabi 频率为 $\Omega(t) = \Omega_0\,(1 + \xi_t)$。OU 过程的稳态方差为 $\langle\xi^2\rangle_{\text{ss}} = \sigma^2/2\theta$，对应相对强度噪声 RIN $\sim 0.014$。

### (4) 激光相位 servo bump

锁相环（PLL）的 servo bump 在功率谱密度中引入特征性的尖峰结构：

$$S_\phi(f)\big|_{\text{peak}} = -80\;\text{dBc/Hz} \tag{5.5}$$

该相位噪声在 servo bandwidth 附近（典型值 $\sim 100\;\text{kHz}$）最为显著，直接导致激发激光的相干性降低，表现为 $|g\rangle$-$|r\rangle$ 跃迁的纯退相（pure dephasing）。在数值模拟中，我们将其等效为在相应频率处叠加的随机相位扰动。

### (5) 里德堡态衰减 + BBR 致退激发

Rydberg 态的自发辐射和 BBR 诱导跃迁导致不可逆的布居损失，以 Lindblad 耗散算符描述：

$$L_{\text{decay}} = \sqrt{\frac{1}{\tau_{\text{eff}}}}\;|g\rangle\langle r| \tag{5.6}$$

相应的 Lindblad 主方程贡献为 $\mathcal{D}[L_{\text{decay}}]\rho = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$。对于 $\tau_{\text{eff}} = 88\;\mu\text{s}$，衰减速率 $\gamma = 1/\tau_{\text{eff}} \approx 11.4\;\text{kHz}$，在亚微秒门操作中对保真度的直接影响约为 $\gamma T_{\text{gate}} \sim 10^{-2}$ 量级。

## 5.3 三组对照场景

为系统性地评估各类控制方法的性能，我们设计三组参数场景，涵盖不同的时间尺度、驱动强度和噪声条件。各场景的具体参数汇总于 Tab.2。

**Tab.2 实验对照场景**

| 场景 | 描述 | $T_{\text{gate}}$ | $\Omega/2\pi$ | 噪声通道 | 预期传统结果 |
|:---:|:---|:---:|:---:|:---:|:---|
| **A** | 长时间 / STIRAP 兼容 | 5 $\mu$s | 0.8 MHz | (1)+(5) | STIRAP $F \lesssim 0.985$ |
| **B (主)** | 短时间 / 全噪声 | 0.3 $\mu$s | 4.6 MHz | 全部 (1)-(5) | GRAPE $F \lesssim 0.97$ |
| **D** | 3 原子 W 态 | 0.5 $\mu$s | 4.6 MHz | 全部 (1)-(5) | 传统方法 $F < 0.95$ |

**场景 B 是本文的主要基准**。它对应 Evered *et al.* 2023 [Evered23] 的实际实验参数区间：亚微秒的门操作时长、全功率 Rabi 驱动、以及完整的噪声环境。在此条件下，门操作时长 $T_{\text{gate}} = 0.3\;\mu\text{s}$ 仅为 Rabi 周期 $2\pi/\Omega \approx 0.22\;\mu\text{s}$ 的约 1.4 倍，留给优化控制的时间窗口极为有限，传统的 GRAPE 算法在全噪声下保真度被限制在 $F \lesssim 0.97$ [DingEnglund25]。

场景 A 为辅助对照：较长的门时间允许绝热方案（如 STIRAP）发挥作用，但 Rydberg 态衰减的累积使其保真度被限制在 $F \lesssim 0.985$。场景 D 将问题推广至三原子 W 态制备，Hilbert 空间维度从 $4 \times 4$ 增长到 $8 \times 8$，控制景观（control landscape）的复杂度显著增加，传统优化方法的保真度进一步下降至 $F < 0.95$。

## 5.4 评估指标

所有方法在统一的指标体系下进行比较。目标态为 \S3 中定义的 Bell 态（场景 A、B）或 W 态（场景 D）：

$$\rho_{\text{tgt}} = |W\rangle\langle W|$$

状态保真度定义为

$$F = \text{Tr}\bigl(\rho(T)\,\rho_{\text{tgt}}\bigr) \tag{5.7}$$

其中 $\rho(T)$ 为系统在门操作结束时刻 $T$ 的密度矩阵。由于噪声的随机性，单次轨迹的保真度本身是随机变量。因此我们对每个场景执行 **1000 次蒙特卡洛轨迹**（Monte Carlo trajectory），报告以下三项统计量：

- **平均保真度** $\langle F\rangle$：1000 次轨迹保真度的算术平均，反映控制方法的典型性能；
- **最差 5% 分位** $F_{05}$：保真度分布的第 5 百分位数，衡量尾部风险（tail risk），对容错量子计算至关重要；
- **标准差** $\sigma_F$：保真度的标准偏差，度量控制鲁棒性。

$F_{05}$ 的引入是因为在实际量子纠错中，最差情形的门保真度（而非平均值）往往决定了逻辑错误率的上界。一个 $\langle F\rangle$ 很高但 $F_{05}$ 较低的方案在实践中可能不如一个 $\langle F\rangle$ 略低但分布更集中的方案有用。

---

**参考文献**

- [Evered23] S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268 (2023).
- [DingEnglund25] Y. Ding, D. Englund *et al.*, arXiv:2504.11737 (2025).

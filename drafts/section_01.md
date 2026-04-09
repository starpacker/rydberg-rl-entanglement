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

以上标度律的物理根源在于 Rydberg 态波函数在空间上的大幅展开。轨道半径 $\langle r \rangle \propto n^{*2}$ 导致偶极矩元同样以 $n^{*2}$ 增长；极化率则涉及偶极矩的平方除以能级间距（$\propto n^{*-3}$），因而标度为 $n^{*7}$；辐射寿命 $\tau_{\text{rad}} \propto n^{*3}$ 源于自发辐射速率与跃迁偶极矩平方和跃迁频率三次方的乘积成正比，二者的 $n^*$ 依赖恰好给出三次方标度 [SWM10, Gallagher94]。

$C_6$ 系数的 $n^{*11}$ 标度可以从二阶微扰论理解：$C_6 \propto d^4/\Delta E$，其中 $d \propto n^{*2}$ 为偶极矩元，$\Delta E \propto n^{*-3}$ 为 Förster 缺陷能，合计给出 $n^{*11}$。Rydberg blockade 半径定义为 $R_b = (C_6/\hbar\Omega)^{1/6}$，故标度为 $n^{*11/6}$ [SWM10]。

总而言之，Rydberg 态独特地结合了长寿命（$\tau \sim n^{*3}$）与强相互作用（$C_6 \sim n^{*11}$），这一组合在自然界的原子态中几乎无可替代——它正是里德堡原子量子计算平台的物理基石。

## 1.3 BBR 与寿命修正

Rydberg 原子的有效寿命不仅受自发辐射限制，还受到黑体辐射（blackbody radiation, BBR）诱导的受激跃迁的显著影响。在有限温度下，有效衰变速率为

$$\frac{1}{\tau_{\text{eff}}} = \frac{1}{\tau_{0\text{K}}} + \Gamma_{\text{BBR}}, \qquad \Gamma_{\text{BBR}} \approx \frac{4\alpha^3 k_B T}{3 n^{*2} \hbar} \tag{1.2}$$

其中 $\tau_{0\text{K}}$ 为零温下的纯辐射寿命（包含自发辐射至所有低能级的贡献），$\alpha$ 为精细结构常数，$k_B T$ 为热能。BBR 速率 $\Gamma_{\text{BBR}}$ 的 $n^{*-2}$ 标度可直观理解为：黑体光谱在 Rydberg 态间距对应的微波频段（$\propto n^{*-3}$）具有充足的光子数，而跃迁偶极矩元 $\propto n^{*2}$，综合效果使得 BBR 诱导的跃迁速率反而随 $n$ 增大而增加（$\Gamma_{\text{BBR}} \propto n^{*-2}$），相对于辐射寿命的 $n^{*3}$ 增长，BBR 在高 $n$ 时成为主导的退激发机制。

Beterov 等人 [Beterov09] 系统地计算了 Rb、Cs 等碱金属 Rydberg 态在室温下的有效寿命。以 Rb $nS$ 态为例：在 $T = 300\;\text{K}$ 下，$n = 70$ 时 $\tau_{\text{eff}} \approx 50\text{–}80\;\mu\text{s}$，远低于零温辐射寿命 $\tau_{0\text{K}} \approx 150\;\mu\text{s}$。这说明对于 $n \gtrsim 50$ 的 Rydberg 态，BBR 对有效寿命的贡献不可忽视。

从量子计算的角度看，有效寿命 $\tau_{\text{eff}}$ 直接限定了量子门操作时间的上界：必须满足

$$T_{\text{gate}} \ll \tau_{\text{eff}}$$

才能保证门操作在退相干前完成。对于典型的 Rydberg 门方案（$n \sim 60\text{–}80$），$\tau_{\text{eff}} \sim 50\text{–}100\;\mu\text{s}$，而单次门操作时间 $T_{\text{gate}} \sim 0.1\text{–}1\;\mu\text{s}$，留有约两个数量级的裕度。然而，在需要多步序列操作或面临其他退相干通道（如多普勒展宽、激光相位噪声）时，BBR 寿命仍是一个不可忽略的系统性限制 [Beterov09, SWM10]。

---

**参考文献**

- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).
- [Gallagher94] T. F. Gallagher, *Rydberg Atoms* (Cambridge University Press, 1994).
- [SA18] N. Šibalić and C. S. Adams, *Rydberg Physics* (IOP Publishing, 2018).
- [Beterov09] I. I. Beterov *et al.*, "Quasiclassical calculations of blackbody-radiation-induced depopulation rates and effective lifetimes of Rydberg nS, nP, and nD alkali-metal atoms with n ≤ 80," *Phys. Rev. A* **79**, 052504 (2009).

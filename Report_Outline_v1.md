# 期中报告大纲 v1
**标题（暂定）**：
> 驯服量子混沌：里德堡原子阵列中基于强化学习的高保真度纠缠态制备
> *Taming Quantum Chaos: High-Fidelity Entanglement Preparation in Rydberg Atom Arrays via Reinforcement Learning*

**目标定位**
- 篇幅：≈10 页正文 + 附录（不计页数）+ 参考文献
- 课程：原子物理学期中报告
- 风格：前沿专题型；物理为主体（≥70%），AI 作为应用收尾（≤30%）
- 数学深度：核心结论给出完整推导；繁琐 / 辅助推导放入附录
- 后续汇报基于本份详细文档抽取讲稿

---

## 0. 整体页数与字数预算

| Part | 主题 | 页数 | 物理 / AI 比重 |
|---|---|---:|---|
| Abstract | 摘要 | 0.3 | — |
| §1 | Rydberg 原子的物理基础（QDT、标度律） | 1.7 | 物理 |
| §2 | 原子—激光相互作用（RWA、双光子激发） | 1.8 | 物理 |
| §3 | 里德堡阻塞与 Bell 态制备 | 2.2 | 物理 |
| §4 | 退相干预算与传统控制算法的失效 | 2.0 | 物理 (含控制论) |
| §5 | **Experiment Setting**：可量化的失效场景 | 0.8 | 物理 |
| §6 | AI 控制：PPO + Domain Randomization | 1.0 | AI |
| §7 | 复现与结果（基于 RL4qcWpc / Plan B） | 0.7 | AI |
| §8 | 讨论、展望与结论 | 0.5 | — |
| 附录 A–D | 详细推导 | (不计) | — |
| 参考文献 | — | (不计) | — |

物理 vs AI ≈ 8.5 : 1.5（页面意义上）。完全符合"原子物理课"的立场。

---

## Part 1 — Rydberg 原子的物理基础（≈1.7 页）

### 1.1 从氢原子到碱金属里德堡原子（0.5 页）
- 类氢能级 $E_n = -\text{Ry}/n^2$ 简短回顾
- **碱金属差异**：内壳层电子的极化与穿透效应 → 价电子在 $r \to 0$ 处感受到非纯库仑势
- **Rydberg–Ritz 公式**（核心结论，需写出）：
  $$ E_{n\ell j} = -\frac{\text{Ry}^*}{(n - \delta_{n\ell j})^2}, \qquad \delta_{n\ell j} = \delta_0 + \frac{\delta_2}{(n-\delta_0)^2} + \cdots $$
- 数值示例（来自 ARC / Šibalić & Adams）：Rb 的 $\delta_0(nS) \approx 3.13$，$\delta_0(nP) \approx 2.65$，$\delta_0(nD) \approx 1.35$
- **附录 A**：放完整的 quantum defect theory 推导（WKB 相位积分 + 短程相移）

### 1.2 标度律（核心，0.7 页）
关键标度律表（必须出现在正文）：

| 物理量 | 标度 | 物理意义 |
|---|---|---|
| 轨道半径 $\langle r\rangle$ | $\sim n^{*2}$ | 原子尺度 → 巨观偶极矩 |
| 偶极矩元 $\langle nS\|er\|nP\rangle$ | $\sim n^{*2}$ | 强光耦合 |
| 极化率 $\alpha$ | $\sim n^{*7}$ | 对外场极敏感 |
| 寿命 $\tau$ (radiative) | $\sim n^{*3}$ | 长寿命 |
| BBR 限寿命 (300 K) | $\sim n^{*2}$ | 室温热辐射不可忽略 |
| vdW 系数 $C_6$ | $\sim n^{*11}$ | 强长程相互作用 |
| 阻塞半径 $R_b = (C_6/\hbar\Omega)^{1/6}$ | $\sim n^{*11/6}$ | 数 μm 量级 |

- 数值锚点：Rb $70S$，$C_6 \approx 862\;\text{GHz}\cdot\mu\text{m}^6$，$\tau_{0\text{K}}\approx 150\;\mu\text{s}$，$\tau_{\text{eff,300K}}\approx 80\;\mu\text{s}$
- 一句话结论：里德堡态独特地兼具 **(i) 长寿命**（足够实现门操作）和 **(ii) 强相互作用**（足够实现量子纠缠）—— 这是其作为量子计算平台的物理根基

### 1.3 BBR 与寿命修正（0.5 页）
- 自发辐射 + 黑体辐射诱导跃迁 (BBR) → 有效寿命
- 写出关键公式：
  $$ \frac{1}{\tau_{\text{eff}}} = \frac{1}{\tau_{0\text{K}}} + \Gamma_{\text{BBR}}, \qquad \Gamma_{\text{BBR}} \approx \frac{4 \alpha^3 k_B T}{3 n^{*2}\hbar} $$
- 引用 Beterov et al. PRA 2009 给出 Rb $nS$ 室温下的具体数值
- **物理意义**：寿命限制了任何门操作的最大允许时长 $T_{\text{gate}} \ll \tau_{\text{eff}}$

**§1 关键引用**：Saffman-Walker-Mølmer RMP 2010（必引）；Gallagher 1994 教科书；Beterov 2009；Šibalić & Adams 2018 / ARC 包

---

## Part 2 — 原子—激光相互作用（≈1.8 页）

### 2.1 二能级原子 + 经典激光场 + RWA（核心推导，0.8 页）
**保留原稿 §1.1 的全部推导但展开**：
- 偶极近似的物理依据：$\lambda \gg a_0$
- 偶极哈密顿量 $\hat H_I = -\hat{\mathbf d}\cdot\mathbf E(t)$
- 引入 bare Rabi 频率 $\Omega = -\mathbf d_{rg}\cdot\mathbf E_0/\hbar$
- 转动坐标变换 $\hat U(t) = e^{-i\omega_L t |r\rangle\langle r|}$
- **RWA 严格判据**：丢弃的快振荡项 $\sim e^{\pm 2i\omega_L t}$ 满足 $|\Omega/\omega_L| \ll 1$（对光学频段成立）
- 最终有效 Hamiltonian（必写）：
  $$ \hat H_{\text{rot}} = -\hbar\Delta\,|r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl(|r\rangle\langle g| + |g\rangle\langle r|\bigr) $$
- Bloch 球图像 + 共振 Rabi 振荡 $P_r(t) = \sin^2(\Omega t/2)$
- AC Stark 位移：$\delta_{\text{AC}} = \Omega^2/(4\Delta)$（小篇幅提一下）

### 2.2 双光子激发到里德堡态（核心，0.7 页）
- 实验现实：单光子直接激发 $5S\to nS$ 的紫外波长难处理 → 用双光子梯形 $5S_{1/2} \to 5P_{3/2} \to nS/nD$
- $\Lambda$ 型三能级 + 大失谐 → 中间态绝热消除
- **推导有效 Rabi 频率**（核心结论，主文给出）：
  $$ \Omega_{\text{eff}} = \frac{\Omega_1\Omega_2}{2\Delta}, \qquad \delta_{\text{AC}} = \frac{\Omega_1^2 - \Omega_2^2}{4\Delta} $$
- 散射代价：$R_{\text{sc}} = \gamma_{5P}\,\Omega_1^2/(4\Delta^2)$，决定 $\Delta$ 的取值权衡
- 数值锚点（de Léséleuc 2018）：$\Omega_1/2\pi \sim 100\;\text{MHz}$, $\Delta/2\pi \sim 740\;\text{MHz}$, $\Omega_{\text{eff}}/2\pi \sim 2\;\text{MHz}$
- **附录 B**：完整的二阶绝热消除推导（投影算符方法 / Schrieffer-Wolff）

### 2.3 偏振、选择定则与 Zeeman 子能级（0.3 页，简短）
- 简短陈述：在严谨实验中需要选定 $|m_J\rangle$ 通道，避免暗态泄漏
- 引用 Walker-Saffman PRA 2008 关于 $m_J$ 平均的 $C_6$ 修正

**§2 关键引用**：Saffman-Walker-Mølmer 2010 §IV；de Léséleuc et al. PRA 2018（最干净的双光子 + RWA 推导）

---

## Part 3 — 里德堡阻塞与 Bell 态制备（≈2.2 页，全报告物理顶峰）

### 3.1 双原子相互作用：从 $1/R^3$ 到 $1/R^6$（0.5 页）
- 偶极—偶极矩阵元 $V_{dd}(R) \propto 1/R^3$
- 当近共振对态远离 → 二阶微扰 → vdW $V_{\text{vdW}} = C_6/R^6$
- **Förster 共振**简介：当 $|nS\rangle + |nS\rangle \leftrightarrow |nP\rangle + |(n-1)P\rangle$ 几乎简并时退化为 $1/R^3$
- 给出 $C_6$ 的物理表达式：
  $$ C_6 = \sum_{\alpha,\beta} \frac{|\langle r r| \hat V_{dd} | \alpha\beta\rangle|^2}{E_{rr} - E_{\alpha\beta}} $$
- **附录 C**：Walker-Saffman 含 Zeeman 简并的完整通道求和

### 3.2 阻塞机制的严格推导（核心，1.0 页）
**完全保留并扩写原稿 §1.2**：
- 双原子总哈密顿量 $\hat H_{\text{tot}}$ 在 $\{|gg\rangle,|gr\rangle,|rg\rangle,|rg\rangle,|rr\rangle\}$ 基下的 4×4 矩阵
- 对称/反对称基变换：$|W\rangle = \tfrac{1}{\sqrt 2}(|gr\rangle+|rg\rangle)$，$|D\rangle = \tfrac{1}{\sqrt 2}(|gr\rangle-|rg\rangle)$
- **关键观察**：$|gg\rangle$ 仅与 $|W\rangle$ 耦合（暗态 $|D\rangle$ 完全脱钩），矩阵元为 $\sqrt{2}\,\hbar\Omega/2$
- **阻塞极限** $V_{\text{vdW}}\gg\hbar\Omega$ → 双激发 $|rr\rangle$ 失谐，整个动力学塌缩到 $|gg\rangle\leftrightarrow|W\rangle$ 的二能级系统：
  $$ \hat H_{\text{eff}} = \frac{\hbar\Omega_{\text{eff}}}{2}\bigl(|gg\rangle\langle W| + |W\rangle\langle gg|\bigr), \qquad \Omega_{\text{eff}} = \sqrt 2\,\Omega $$
- $\sqrt 2$ 增强是 Bell 态实验的"指纹"（Gaëtan 2009 / Urban 2009 实验观察到）
- **阻塞半径**定义：
  $$ R_b \;\equiv\; \left(\frac{C_6}{\hbar\Omega}\right)^{1/6} $$
  并给出 Rb 70S, $\Omega/2\pi=1$ MHz 时 $R_b\approx 9.7$ μm 的具体数值

### 3.3 完美 Bell 态产生协议（0.4 页）
- 时间 $t = \pi/\Omega_{\text{eff}}$ 时 $|gg\rangle\to -i|W\rangle$ —— 极大纠缠的 Bell 态
- 概念性扩展：
  - $N$ 原子推广 → $\sqrt N$ 增强 + W 态制备
  - 两段式 Levine 协议（off-resonant + global） → CZ 门
- 提及当前最佳实验：Levine 2019 ($F_{\text{Bell}}\geq 0.950$, $F_{\text{CZ}}\geq 0.974$)，Evered 2023 ($F_{\text{CZ}} = 0.9952$)，Scholl 2023 ($F_{\text{Bell}}=0.9971$)

### 3.4 实验平台简述（0.3 页）
- 光镊阵列 (SLM/AOD)，缺陷无关的 atom-by-atom 装填
- 引用 Browaeys-Lahaye Nature Phys 2020、Endres Science 2016、Bernien Nature 2017

**§3 关键引用**：Lukin 2001 PRL（必引，原始提案）；Urban 2009 / Gaëtan 2009 Nat Phys（首次实验）；Walker-Saffman 2008 PRA；Levine 2019 PRL；Browaeys-Lahaye 2020 综述

---

## Part 4 — 退相干预算与传统算法的失效（≈2.0 页）

### 4.1 退相干来源的全景图（0.7 页）
逐项列出，并给出影响的数学形式：

| 来源 | 数学描述 | 典型值 (Rb 53S, T=10 μK) | 引用 |
|---|---|---|---|
| 自发辐射 + BBR | Lindblad $L_k = \sqrt{\Gamma_r}\|g\rangle\langle r\|$ | $\tau_{\text{eff}} \approx 88\;\mu\text{s}$ | Beterov 2009 |
| 中间态散射 ($5P/6P$) | $R_{\text{sc}} = \gamma_{5P}\Omega_1^2/(4\Delta^2)$ | $\sim 10^{-4}/\mu\text{s}$ | Evered 2023 |
| 多普勒位移 | $\delta\omega = \mathbf k_{\text{eff}}\cdot\mathbf v$, $\sigma_v = \sqrt{k_B T/m}$ | $\sigma_\omega/2\pi \approx 50\;\text{kHz}$ | de Léséleuc 2018 |
| 位置抖动 ($1/R^6$ 灵敏) | $\delta V = -6 V_{\text{vdW}}\delta R/R$ | $\sigma_R \approx 100\;\text{nm}$ | de Léséleuc 2018 |
| 激光强度 OU 噪声 | $\Omega(t) = \Omega_0(1+\xi(t))$ | $\delta\Omega/\Omega \sim 1\text{-}2\%$ | Day 2022 |
| 激光相位噪声 (servo bump) | PSD $S_\phi(f)$ 在 $f\sim\Omega$ 处尖峰 | $-80\;\text{dBc/Hz}$ → $10^{-3}/\text{gate}$ | Day 2022; PRX Quantum 2025 |

### 4.2 保真度的线性响应理论（核心，0.5 页）
- 写出 Day-Bohnet-Schleier-Smith / PRX Quantum 2025 的线性响应公式：
  $$ 1 - F \;\approx\; \sum_j \int_0^\infty df\, S_j(f)\,I_j(f) $$
  其中 $S_j(f)$ 是噪声功率谱，$I_j(f)$ 是控制脉冲的灵敏度滤波函数
- 标度律小结（这是论证传统脉冲被限制的关键）：
  - 频率噪声：$1-F \propto \Omega^{-1.79}$
  - 里德堡退相干：$1-F \propto \Omega^{-1}$
  - 多普勒：$1-F \propto \Omega^{-2}$
- **物理解读**：要降低这些误差就要把 $\Omega$ 调大；但 $\Omega$ 太大又破坏了阻塞条件 $V_{\text{vdW}}\gg\hbar\Omega$ —— 物理上的双重夹击

### 4.3 传统控制算法及其失效模式（0.8 页）
不展开公式，重点描述失效机制（每段一段话 + 一条公式）：

#### (a) STIRAP / 绝热协议
- 暗态 $|D(t)\rangle = \cos\theta(t)|g\rangle - \sin\theta(t)|r\rangle$
- **adiabatic speed limit**：$|\dot\theta|\ll\Omega_{\text{eff}}$，要求 $\Omega T \gtrsim 10$
- **Goldilocks 区间**：$T_{\text{QSL}}\ll T \ll \tau_{\text{Ryd}}$
- 数值证据：在 $\Omega/2\pi < 1\;\text{MHz}$ 的 STIRAP 兼容区域，多普勒贡献 $> 2\times 10^{-3}$，里德堡衰减贡献 $> 10^{-2}$ → STIRAP 保真度被天花板锁在 $F \lesssim 0.985$
- 引用：Vitanov RMP 2017；Sun-Robicheaux arXiv:1912.02977；Yagüe Bosch 2023

#### (b) GRAPE (Gradient Ascent Pulse Engineering)
- 一句话写出梯度公式（不展开推导）：
  $$ \frac{\partial \Phi}{\partial u_k(t_j)} = -i\Delta t\,\langle\psi_{\text{tgt}}| U_N\cdots [H_k, U_j]\cdots U_1|\psi_0\rangle + \text{c.c.} $$
- **致命缺陷 1：Sim2Real Gap**。GRAPE 寻找的最优脉冲利用了 $H_{\text{sim}}$ 的精细干涉路径；当 $H_{\text{real}} = H_{\text{sim}} + \delta H$ 时，干涉破坏 → 保真度雪崩
- **致命缺陷 2：高维 barren plateau**。Larocca et al. Quantum 2022 证明高维控制空间的梯度指数消失
- 数值证据：Goerz 2014 显示 Krotov/GRAPE 脉冲在容差窗口外保真度灾难性下降
- **附录 D**：放 GRAPE 梯度的完整推导

#### (c) 反绝热驱动 (Counterdiabatic / Shortcuts to Adiabaticity)
- Berry 公式 $H_{\text{CD}} = i\hbar\sum_n |\partial_t n\rangle\langle n|$
- **失效原因**：CD 项是从闭系瞬时本征基底导出的，开系下暗态已经不是动力学本征态。**有色噪声**（1/f Doppler、激光 servo bump）引入的非马尔可夫失配无法被任何局域反项补偿
- 引用：Yagüe Bosch 2023（其 STA-CZ 方案明确不含开系建模）

### 4.4 小结：为什么我们需要新工具（0.0 页，融入 4.3 末段）
传统算法都建立在"已知精确 Hamiltonian + 闭系或马尔可夫开系"的假设之上。一旦面对**未知噪声谱 + 强非线性参数依赖（$1/R^6$）+ 多源失配**的真实实验，它们都会失效。这正是把控制问题重新框定为**无模型强化学习**问题的物理动机。

**§4 关键引用**：Vitanov RMP 2017；Khaneja JMR 2005；Goerz 2014 PRA；Larocca Quantum 2022；Day npj QI 2022；PRX Quantum 6, 010331 (2025) [arXiv:2407.20184]；Evered Nature 2023；Yagüe Bosch 2023

---

## Part 5 — Experiment Setting：可量化的失效场景（≈0.8 页，关键章节）

> **目的**：构造一个**严格、可复现、参数明确**的实验设置，使得在该设置下传统算法（STIRAP / GRAPE / STA）可被定量地证明失效，并为后续 RL 训练提供一致的基准。

### 5.1 物理系统
- $^{87}$Rb 双原子，囚禁于光镊
- 里德堡态：$|r\rangle = |53S_{1/2}, m_J=+1/2\rangle$，$\tau_{\text{eff}} = 88\;\mu\text{s}$
- 双光子激发 via $6P_{3/2}$，$\Delta/2\pi = 7.8\;\text{GHz}$
- 原子间距 $R = 2\;\mu\text{m}$，$V_{\text{vdW}}/2\pi \approx 500\;\text{MHz}$
- 基线 Rabi 频率 $\Omega/2\pi = 4.6\;\text{MHz}$（与 Evered 2023 一致）

### 5.2 噪声模型 (Lindblad + 经典随机)
1. **多普勒**：$\delta_i \sim \mathcal N(0,\sigma_D^2)$，$\sigma_D/2\pi = 50\;\text{kHz}$ ($T=10\;\mu\text{K}$)
2. **位置抖动**：$\delta R \sim \mathcal N(0, \sigma_R^2)$，$\sigma_R = 100\;\text{nm}$ → $1/R^6$ 非线性放大
3. **激光幅度 OU 噪声**：$d\xi_t = -\theta\xi_t\,dt + \sigma\,dW_t$，$\theta^{-1}=10\;\mu\text{s}$，$\sigma = 0.02$
4. **相位 servo bump**：$S_\phi(f)$ 在 $f=\Omega/2\pi$ 处尖峰高度 $-80\;\text{dBc/Hz}$
5. **里德堡衰减 + BBR**：Lindblad 算符 $\sqrt{1/\tau_{\text{eff}}}\,|g\rangle\langle r|$

### 5.3 三组对照场景
| 场景 | $T_{\text{gate}}$ | $\Omega/2\pi$ | 噪声 | 预期传统结果 |
|---|---|---|---|---|
| **A. 长时间 / STIRAP 兼容** | $5\;\mu\text{s}$ | $0.8\;\text{MHz}$ | 仅 (1)+(5) | STIRAP $F \lesssim 0.985$（绝热可行但被衰减锁死） |
| **B. 短时间 / GRAPE 区** | $0.3\;\mu\text{s}$ | $4.6\;\text{MHz}$ | 全部 (1)–(5) | GRAPE $F \lesssim 0.97$（servo bump + 幅度漂移雪崩） |
| **C. 强随机 / 实战区** | $1\;\mu\text{s}$ | $4.6\;\text{MHz}$ | 全部 (1)–(5) + 参数 ±10% 漂移 | GRAPE/STA 双双 < 0.95 |

### 5.4 评估指标
- 终态保真度 $F = \text{Tr}(\rho(T)\rho_{\text{tgt}})$，目标 $\rho_{\text{tgt}} = |W\rangle\langle W|$
- 在 1000 次蒙特卡洛轨迹下的均值 $\langle F\rangle$、最差 5% 分位 $F_{05}$、稳健性 $\sigma_F$
- 这三个指标后续给 PPO 用同样的设置训练 + 测试

---

## Part 6 — AI 控制：PPO + Domain Randomization（≈1.0 页）

### 6.1 把开系量子动力学映射成 MDP（0.3 页）
- 状态空间 $\mathcal S$：密度矩阵 $\rho(t)$ 的实向量化（4×4 复矩阵 → 30 维实向量）
- 动作空间 $\mathcal A$：$a_t = (\Omega(t), \Delta(t)) \in \mathbb R^2$
- 转移：Lindblad 主方程
  $$ \dot\rho = -i[H(t),\rho] + \sum_k\bigl(L_k\rho L_k^\dagger - \tfrac{1}{2}\{L_k^\dagger L_k,\rho\}\bigr) $$
- 奖励：稀疏终态 $r_T = \text{Tr}(\rho(T)\rho_{\text{tgt}})$（或 $r_T = -\log(1-F)$ 加速学习）

### 6.2 PPO 算法核心（0.4 页）
- 写出 clipped surrogate objective（必含）：
  $$ L^{\text{CLIP}}(\theta) = \hat{\mathbb E}_t\!\Bigl[\min\!\bigl(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\bigr)\Bigr] $$
- **为什么是 PPO 不是 DDPG/SAC**：
  - 量子保真度景观 spin-glass 化（Bukov PRX 2018）
  - 离策略方法的 replay buffer 放大 Q 值 bootstrap 错误
  - **直接经验证据**：Ernst et al. ICML 2025 (arXiv:2501.14372) 在里德堡 CZ 上对比 PPO/DDPG/TD3，PPO 平均保真度领先一个数量级以上

### 6.3 Domain Randomization 的物理意义（0.3 页）
- 训练时每个 episode 从 §5.2 的噪声分布**重新采样**：$\sigma_D, \sigma_R, \xi(t), $ servo bump 高度
- 神经网络学到的**不是某条最优脉冲**，而是**一个对噪声分布鲁棒的反馈策略**
- 物理类比：相当于学习一个"动态退耦序列"，通过相位 chirp 和幅度调制把系统与噪声谱解耦

**§6 关键引用**：Bukov PRX 2018；Niu et al. npj QI 2019（首个量子控制 + 噪声训练）；Ernst et al. arXiv:2501.14372（PPO vs DDPG/TD3 直接对比，里德堡 CZ）；Ding-Englund arXiv:2504.11737；Guatto et al. Nat Commun 2024（实验 sim-to-real）

---

## Part 7 — 复现与结果（≈0.7 页）

### 7.1 代码实现路线（双轨）

**Plan A — 基于 RL4qcWpc**
- 仓库：https://github.com/jan-o-e/RL4qcWpc  （对应 Ernst et al. ICML 2025）
- 栈：JAX + Gymnax + Qiskit-Dynamics + Distrax
- 内置里德堡 CZ 环境，论文宣称 $F=0.9996$
- 风险：JAX-CUDA 在 Windows 上不友好；建议 WSL2 或 CPU 运行小规模

**Plan B — 自建轻量栈（推荐 fallback，≈300 行）**
- 物理后端：`qutip` + `krotov`（krotov 文档自带"两个里德堡原子 CPHASE + 幅度/能级涨落"示例 → 直接复用 Hamiltonian 模型）
- 环境包装：`gymnasium.Env`，`step()` 调用 QuTiP `mesolve` 演化一个时间片
- RL 训练：`stable-baselines3` 的 PPO（默认超参数即可）
- 基线对照：`qutip-qtrl` 的 GRAPE / Krotov（用于画 RL vs 传统的对比曲线）
- 优势：纯 Python，Windows 友好，笔记本 CPU 几小时收敛

### 7.2 计划呈现的结果图表
1. **图 1**：阻塞机制示意 + Rb 53S 能级图（手画/示意）
2. **图 2**：场景 A/B/C 下，STIRAP/GRAPE/PPO 三种方法的 $\langle F\rangle$ 对比柱状图（核心结果）
3. **图 3**：保真度随时间漂移参数 $\delta\Omega/\Omega \in [0,5\%]$ 的鲁棒性曲线
4. **图 4**：PPO 训练学习曲线（reward vs episodes，3 个 seed）
5. **图 5**：PPO 策略输出的 $(\Omega(t), \Delta(t))$ 脉冲形状，对比 GRAPE 脉冲（直观看出 RL 偏好 chirp + smooth amplitude）

### 7.3 诚实的可行性声明
- 我们 **不期望** 一次跑出 0.9996——目标是定性证明 PPO + DR 在场景 C 下保真度显著高于 GRAPE / STIRAP
- 复现报告中明确写明"在场景 X 下，PPO 达到 $F = 0.YY$，GRAPE 仅 $F = 0.ZZ$"

---

## Part 8 — 讨论、展望与结论（≈0.5 页）

- **已完成的物理证明链**：从单原子 RWA → 双原子阻塞 → Bell 态 → 实际退相干 → 传统控制的失效边界
- **AI 的位置**：不是"魔法"，而是一种**无需显式梯度、对噪声分布泛化**的控制策略
- **局限性**：
  - 当前 RL 仅做 2-原子；多原子扩展面临状态空间 $4^N$ 维灾难
  - 训练分布与真实噪声分布的失配仍是 sim-to-real 主要瓶颈（Guatto 2024）
- **展望**：
  - 物理：Sr/Yb 碱土类里德堡门 + erasure conversion (Scholl 2023)
  - 算法：基于量子 Fisher 信息的内在奖励、可微量子模拟器 + actor-critic 混合

---

## 附录（不计页数，保留全部细致推导）

| 附录 | 内容 |
|---|---|
| **A** | Quantum defect theory 的 WKB / 相位积分推导，量子亏损系数的物理意义 |
| **B** | 双光子 $\Lambda$ 系统中间态的二阶绝热消除（投影算符 / Schrieffer-Wolff） |
| **C** | Walker-Saffman 含 Zeeman 简并的 $C_6$ 通道求和与 Förster 共振分析 |
| **D** | GRAPE 梯度公式的完整推导 + Larocca 类型 barren plateau 简短证明 |
| **E** | Lindblad 主方程的 Kraus 算符表示与数值积分细节 |
| **F**（选） | PPO 完整算法伪代码 + 关键超参数表 |

---

## 参考文献清单（按主题分组，所有 arXiv 编号已核对）

**A. 里德堡物理基础**
- [SWM10] Saffman, Walker, Mølmer, *Rev. Mod. Phys.* **82**, 2313 (2010). arXiv:0909.4777
- [Gallagher94] Gallagher, *Rydberg Atoms*, Cambridge (1994)
- [SA18] Šibalić & Adams, *Rydberg Physics*, IOP (2018)
- [Beterov09] Beterov et al., *PRA* **79**, 052504 (2009). arXiv:0810.0339

**B. 原子—激光相互作用**
- [dL18] de Léséleuc et al., *PRA* **97**, 053803 (2018). arXiv:1802.10424

**C. 阻塞与实验**
- [Lukin01] Lukin et al., *PRL* **87**, 037901 (2001). arXiv:quant-ph/0011028
- [Urban09] Urban et al., *Nat. Phys.* **5**, 110 (2009). arXiv:0805.0758
- [Gaetan09] Gaëtan et al., *Nat. Phys.* **5**, 115 (2009). arXiv:0810.2960
- [WS08] Walker & Saffman, *PRA* **77**, 032723 (2008). arXiv:0712.3438
- [Levine19] Levine et al., *PRL* **123**, 170503 (2019). arXiv:1908.06101
- [Madjarov20] Madjarov et al., *Nat. Phys.* **16**, 857 (2020). arXiv:2001.04455
- [Evered23] Evered et al., *Nature* **622**, 268 (2023). arXiv:2304.05420
- [Scholl23] Scholl et al., *Nature* **622**, 273 (2023). arXiv:2305.03406
- [BL20] Browaeys & Lahaye, *Nat. Phys.* **16**, 132 (2020). arXiv:2002.07413
- [Endres16] Endres et al., *Science* **354**, 1024 (2016). arXiv:1607.03044
- [Bernien17] Bernien et al., *Nature* **551**, 579 (2017). arXiv:1707.04344

**D. 传统量子控制**
- [Vitanov17] Vitanov et al., *Rev. Mod. Phys.* **89**, 015006 (2017). arXiv:1605.00224
- [Moller08] Møller, Madsen, Mølmer, *PRL* **100**, 170504 (2008). arXiv:0802.3631
- [Khaneja05] Khaneja et al., *J. Magn. Reson.* **172**, 296 (2005)
- [Caneva11] Caneva, Calarco, Montangero, *PRA* **84**, 022326 (2011). arXiv:1103.0855
- [Goerz14] Goerz et al., *PRA* **90**, 032329 (2014). arXiv:1401.1858
- [Berry09] Berry, *J. Phys. A* **42**, 365303 (2009)
- [YB23] Yagüe Bosch et al., *Ann. Phys.* (2023). arXiv:2312.11594
- [Day22] Day, Ramette, Schleier-Smith, *npj QI* **8**, 72 (2022). arXiv:2112.04946
- [PRXQ25] *PRX Quantum* **6**, 010331 (2025). arXiv:2407.20184
- [Larocca22] Larocca et al., *Quantum* **6**, 824 (2022). arXiv:2105.14377

**E. RL & AI 控制**
- [Bukov18] Bukov et al., *PRX* **8**, 031086 (2018). arXiv:1705.00565
- [Fosel18] Fösel et al., *PRX* **8**, 031084 (2018). arXiv:1802.05267
- [Niu19] Niu et al., *npj QI* **5**, 33 (2019). arXiv:1803.01857
- [Ernst25] Ernst et al., ICML 2025. arXiv:2501.14372 ★ (核心引用 + 配套代码)
- [DingEnglund25] Ding & Englund, arXiv:2504.11737 (2025)
- [Guatto24] Guatto et al., *Nat. Commun.* **15**, 8901 (2024)
- [PPO17] Schulman et al., "Proximal Policy Optimization", arXiv:1707.06347

**F. 代码 / 工具**
- [RL4qcWpc] https://github.com/jan-o-e/RL4qcWpc
- [krotov] https://github.com/qucontrol/krotov
- [qutip-qtrl] https://github.com/qutip/qutip-qtrl
- [ARC] https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator
- [SB3] https://github.com/DLR-RM/stable-baselines3

---

## 待你确认的设计决策（✅ / ❌ / 修改）

1. **页数与结构分配**：§3 (阻塞) 给 2.2 页是物理高潮；§5 (Experiment Setting) 单列 0.8 页 —— 你是否同意？
2. **附录策略**：QDT、绝热消除、Walker-Saffman $C_6$、GRAPE 梯度全部下沉到附录 —— 同意吗？还是想把 QDT / 绝热消除 提到正文？
3. **实验设置 §5**：场景 A/B/C 三组的物理参数（基于 Evered 2023 的真实数字）你觉得合适吗？是否需要再加场景 D（多原子推广）？
4. **复现路线**：先尝试 Plan A (RL4qcWpc) 还是直接走 Plan B (krotov + SB3)？建议直接 Plan B，原因：(a) Windows 友好，(b) 你完全理解每一行代码，oral defense 时讲得清楚。
5. **AI 部分页数**：当前 §6+§7 共 1.7 页 —— 是否还想压缩到 1.5 页内？或反之希望扩到 2 页？
6. **图表数量**：目前规划 5 张图 —— 是否能在你的时间预算内完成（图 2、3 是必须，图 4、5 是加分项）？
7. **是否要在大纲最后给出一份"每周时间线"** —— 例如 Week 1 复现 / Week 2 训练 / Week 3 写作？

请逐条回应，我会基于你的反馈把这份大纲推进到 v2，然后我们就可以开始按节填充正文了。

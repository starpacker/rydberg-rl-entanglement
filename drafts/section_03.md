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

在实际量子计算中，我们更需要的是 controlled-Z (CZ) 门而非单纯的 Bell 态制备。Levine 等人 [Levine19] 提出了一个优雅的两段式协议：

1. **第一段**：对控制原子施加 $\pi$ 脉冲（$|g\rangle_c \to |r\rangle_c$，或 $|r\rangle_c \to |g\rangle_c$），条件性地阻塞目标原子；
2. **第二段**：对目标原子施加 $2\pi$ 脉冲——若未被阻塞，则获得一个 $\pi$ 几何相位；若被阻塞，则无变化。

整体效应为 $|rr\rangle \to -|rr\rangle$，其余 basis 不变——精确实现了 CZ 门。

实验上，Bell 态保真度的记录不断刷新：

- Levine *et al.* 2019 [Levine19]：$F_{\text{Bell}} \geq 0.950$，首次超过纠缠态 percolation 阈值；
- Evered *et al.* 2023 [Evered23]：$F_{\text{CZ}} = 0.9952$，基于 $^{87}\text{Rb}$ $60S$ 态，在 60 个原子的阵列上实现；
- Scholl *et al.* 2023 [Scholl23]：$F_{\text{Bell}} = 0.9971$，基于 $^{87}\text{Rb}$ 碱土类方案（erasure conversion），达到迄今最高 Bell 态保真度。

从 $F \approx 0.95$ 到 $F > 0.995$ 的进步充分表明 Rydberg blockade 机制在工程上的可扩展性。然而，进一步逼近容错量子计算所需的 $F > 0.999$ 门槛，要求对退相干源进行精细的控制优化——这正是 \S4 和后续章节将要讨论的挑战。

## 3.4 实验平台简述

Rydberg blockade 门的实验实现依赖于对单个中性原子的精确囚禁和排列。近十年来，**光镊阵列**（optical tweezer array）技术的发展为此提供了理想的平台。

### 光镊阵列与无缺陷装填

单个光镊由高数值孔径物镜聚焦的激光束形成，焦点处的势阱深度约 $\sim 1\;\text{mK}$（以 $k_B T$ 为单位），足以囚禁从磁光阱（MOT）中加载的冷原子。利用空间光调制器（SLM）或声光偏转器（AOD），可以在焦平面上生成任意几何构型的光镊阵列——从一维链到二维方阵、三角晶格乃至三维结构。

由于 MOT 加载是随机过程，初始阵列中各光镊的占据为随机的（典型占据率约 50\%）。Endres 等人 [Endres16] 发展了一套基于实时荧光成像和 AOD 动态重排的 atom-by-atom assembly 技术：首先对阵列进行成像以确定各位点的占据情况，然后用额外的"移动光镊"将原子从有余的位点搬运到空缺处，最终实现无缺陷（defect-free）的确定性阵列。该技术已被推广到数百个原子的规模 [Bernien17, BL20]。

### 从阵列到量子操控

一旦原子被确定性地排列在光镊阵列中，Rydberg 激发和量子门操作遵循 \S2 中描述的双光子方案：全局或寻址的激光脉冲将选定的原子对激发到 Rydberg 态，利用 blockade 机制实现纠缠。Browaeys 和 Lahaye 的综述 [BL20] 全面介绍了该平台从物理原理到量子模拟和量子计算应用的完整图景。

光镊阵列平台的核心优势在于：(i) 高度可编程的原子几何构型，(ii) 原子间距可在 $2\text{--}10\;\mu\text{m}$ 范围内灵活调节——恰好覆盖典型的 blockade 半径 $R_b$，(iii) 可扩展至 $>1000$ 个量子比特 [BL20]。这些特性使其成为当前最有前景的中性原子量子计算平台之一。

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
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

### Bare Rabi frequency

定义 **bare Rabi frequency**

$$
\Omega = -\frac{\mathbf{d}_{rg} \cdot \mathbf{E}_0}{\hbar},
$$

其中 $\mathbf{d}_{rg} = \langle r|\hat{\mathbf{d}}|g\rangle$ 为跃迁偶极矩元。总 Hamiltonian 在 Schr\"{o}dinger picture 下为

$$
\hat{H} = \hbar\omega_0 |r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl(e^{-i\omega_L t} + e^{i\omega_L t}\bigr)\bigl(|r\rangle\langle g| + |g\rangle\langle r|\bigr).
$$

### 旋转坐标变换

引入 **rotating frame** 变换

$$
\hat{U}(t) = e^{-i\omega_L t\,|r\rangle\langle r|},
$$

将态矢量变换为 $|\tilde{\psi}(t)\rangle = \hat{U}^\dagger(t)|\psi(t)\rangle$。对应的变换后 Hamiltonian 为

$$
\hat{H}_{\text{rot}} = \hat{U}^\dagger \hat{H} \hat{U} - i\hbar\,\hat{U}^\dagger \dot{\hat{U}}.
$$

展开后得到四项：两项以频率 $\omega_0 - \omega_L$ 缓变（near-resonant terms），两项以频率 $\omega_0 + \omega_L \approx 2\omega_L$ 快速振荡（counter-rotating terms）。

### 旋转波近似 (RWA)

当 **bare Rabi frequency** 远小于光学频率，即

$$
\left|\frac{\Omega}{\omega_L}\right| \ll 1,
$$

快振荡项 $\sim e^{\pm 2i\omega_L t}$ 在任何实验可分辨的时间尺度上平均为零，可安全丢弃。对于光学跃迁（$\omega_L/2\pi \sim 10^{14}\;\text{Hz}$, $\Omega/2\pi \sim \text{MHz}$），该条件以极大余量满足。丢弃 counter-rotating terms 后，定义 detuning $\Delta = \omega_L - \omega_0$，得到 **RWA 有效 Hamiltonian**：

$$
\hat{H}_{\text{rot}} = -\hbar\Delta\,|r\rangle\langle r| + \frac{\hbar\Omega}{2}\bigl(|r\rangle\langle g| + |g\rangle\langle r|\bigr). \tag{2.1}
$$

这是一个时间无关的二能级 Hamiltonian，完全由两个实验可调参数 $(\Omega, \Delta)$ 控制。

### Bloch 球图像

式 (2.1) 的动力学可映射到 Bloch 球上的旋转：定义 Bloch 向量 $\mathbf{B} = (\text{Re}\,\rho_{gr},\;\text{Im}\,\rho_{gr},\;(P_g - P_r)/2)$，则运动方程为 $\dot{\mathbf{B}} = \boldsymbol{\Omega}_{\text{eff}} \times \mathbf{B}$，其中有效磁场 $\boldsymbol{\Omega}_{\text{eff}} = (\Omega, 0, -\Delta)$。态矢量绕 $\boldsymbol{\Omega}_{\text{eff}}$ 方向进动，进动频率为广义 Rabi 频率 $\Omega' = \sqrt{\Omega^2 + \Delta^2}$（参见 Fig. 4）。

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

在大失谐极限 $\Delta \gg \Omega_1, \Omega_2$ 下，中间态 $|e\rangle$ 的布居极小，可被绝热消除（adiabatic elimination）。核心思路是：在三能级的振幅方程中，设 $\dot{c}_e \approx 0$（等价地，投影掉快变子空间），将 $c_e$ 用 $c_g, c_r$ 表示后代入，得到一个有效的二能级系统。其 **effective Rabi frequency** 和 **differential AC Stark shift** 分别为

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

以 de L\'{e}s\'{e}leuc 等人 [dL18] 的实验参数为例：$\Omega_1/2\pi \sim 100\;\text{MHz}$, $\Delta/2\pi \sim 740\;\text{MHz}$, 由此得到 $\Omega_{\text{eff}}/2\pi \sim 2\;\text{MHz}$。此时散射速率 $R_{\text{sc}} \sim \gamma_{5P} \times (100/740)^2/4 \approx 27\;\text{kHz}$，在典型门时间 $T_{\text{gate}} \sim 0.5\;\mu\text{s}$ 内散射概率约 $1.4\%$——这正是推动实验组选择更大失谐（如 $\Delta/2\pi \sim 7.8\;\text{GHz}$ via $6P_{3/2}$, [Evered23]）的动机之一。

---

## 2.3 偏振、选择定则与 Zeeman 子能级

在实际实验中，原子的 Zeeman 子能级 $|m_J\rangle$ 不可忽略。双光子激发的 selection rules 要求 $\Delta m_J = 0, \pm 1$（取决于激光偏振），因此必须精心选择偏振配置以确保仅驱动目标 $|m_J\rangle$ 通道，避免 **dark-state leakage**——即部分布居被泵浦到不参与有效二能级动力学的 Zeeman 子能级中，导致保真度下降。

在计算双原子 van der Waals 系数 $C_6$ 时，若不对 $m_J$ 做精确选择，则需要对所有可能的 Zeeman 通道求和取平均。Walker 与 Saffman [WS08] 给出了 $m_J$-averaged $C_6$ 的系统计算方法，指出不同 $|m_J\rangle$ 通道的 $C_6$ 可差异数倍，这对阻塞半径 $R_b$ 的精确估计至关重要。

---

**本节关键引用**：[SWM10] Saffman, Walker & M\o{}lmer, RMP 2010, \S IV; [dL18] de L\'{e}s\'{e}leuc et al., PRA 2018; [WS08] Walker & Saffman, PRA 2008.

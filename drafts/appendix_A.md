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

其中 $\text{Ry}^* = \text{Ry} \cdot m^*/m_e$ 是修正的 Rydberg constant（考虑 reduced mass 效应），$n^* = n - \delta_\ell$ 称为 effective quantum number。

## A.6 Quantum Defect 的能量依赖性

虽然 $\delta_\ell$ 主要取决于 core 区域的势，但它对能量有弱的依赖性。Ritz 展开将 quantum defect 表示为 effective quantum number 的幂级数：

$$\delta_{n\ell j} = \delta_0 + \frac{\delta_2}{(n-\delta_0)^2} + \frac{\delta_4}{(n-\delta_0)^4} + \cdots \tag{A.3}$$

各项的物理起源：

- **$\delta_0$（主项）**：捕获 valence electron 对 core 的 penetration effect。这是 quantum defect 的主要贡献，由 core 区域 $V(r)$ 与 Coulomb 势的偏差大小决定。对于 Rb 的 $nS$ 态，$\delta_0 \approx 3.13$；对于 $nD$ 态，$\delta_0 \approx 1.35$ [Gallagher94]。

- **$\delta_2$（二阶修正）**：主要来自 ionic core 的 polarization effect。当 valence electron 距离 core 较近时，其电场会极化 core electron cloud，产生 induced dipole moment，进而修正有效势。此效应引入 $\sim -\alpha_d/(2r^4)$ 的极化势，其中 $\alpha_d$ 为 core 的 dipole polarizability。

- **$\delta_4$ 及更高阶项**：来自 quadrupole polarization、exchange interaction 以及 relativistic correction 等更精细的效应。在实际拟合中，通常只需保留到 $\delta_2$ 即可描述现有实验精度 [SA18]。

$\delta_{n\ell j}$ 中 $j$ 的依赖性来自 spin-orbit coupling，对于 $\ell \geq 1$ 的态，fine structure splitting 导致 $j = \ell \pm 1/2$ 的 quantum defect 略有不同。

## A.7 物理图像与 $\ell$ 依赖性

Quantum defect 的大小直接反映 valence electron 对 core 的穿透程度：

- **$\ell = 0$（S 态）**：无 centrifugal barrier，valence electron 的径向波函数在 $r = 0$ 处有限（$u(r) \sim r$），可以深入 core 区域。因此 $\delta_0$ 很大。以 $^{87}\text{Rb}$ 为例，$\delta_0(S_{1/2}) \approx 3.131$ [SA18]。

- **$\ell = 1$（P 态）**：有较弱的 centrifugal barrier $\sim \ell(\ell+1)/r^2$，但 valence electron 仍能显著穿透 core。$\delta_0(P_{1/2}) \approx 2.654$，$\delta_0(P_{3/2}) \approx 2.642$。

- **$\ell = 2$（D 态）**：centrifugal barrier 更强，穿透减小。$\delta_0(D_{3/2}) \approx 1.348$，$\delta_0(D_{5/2}) \approx 1.346$。

- **$\ell \geq 3$（F 态及更高）**：centrifugal barrier 足够高，几乎完全阻止 valence electron 进入 core 区域。$\delta_\ell \ll 1$，能级几乎是 hydrogen-like 的。此时主要的修正来自 long-range polarization 势而非 core penetration。

这一 $\ell$ 依赖性对 Rydberg atom 的实验选择具有重要意义：低 $\ell$ 态的能级间距可被精确测量，用于确定 quantum defect parameters [SWM10]；而高 $\ell$ 态由于接近 hydrogenic，其性质可以用解析公式直接计算。

## A.8 小结

Quantum defect theory 提供了一个优雅而实用的框架，将复杂的多电子 alkali atom 问题归结为仅含少数参数的 modified hydrogen model。式 (A.2) 和 (A.3) 构成了精确计算 Rydberg 态能级的基础，进而决定了 transition frequencies、dipole matrix elements、以及 atom-atom interaction 的强度（参见附录 C）。该理论的成功源于一个基本事实：Rydberg electron 的波函数绝大部分分布在 core 外的 Coulomb 区域，core 内的复杂多体效应仅以少数 phase-shift 参数体现。

---

**参考文献**

- [Gallagher94] T. F. Gallagher, *Rydberg Atoms*, Cambridge University Press (1994).
- [SA18] D. A. Steck, "Rubidium 87 D Line Data," available online (2018).
- [SWM10] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," Rev. Mod. Phys. **82**, 2313 (2010).

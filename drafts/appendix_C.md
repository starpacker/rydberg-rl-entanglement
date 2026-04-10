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

# Task Cards: 里德堡 RL 报告 Sub-Agent 任务分解

> **对应 Spec**: `docs/superpowers/specs/2026-04-09-rydberg-rl-report-design.md`
> **调度方式**: Main Agent 按 Phase 顺序分发，同 Phase 内的 Task Card 可并行执行

---

## 符号约定（所有 Agent 共用）

```
Ry*     = 修正 Rydberg 常数 (考虑 reduced mass)
n*      = n - δ_{nℓj}, 有效量子数
δ_{nℓj} = quantum defect
Ω       = Rabi 频率 (rad/s)，注意 2π 因子
Δ       = 失谐 (rad/s)
V_vdW   = C_6/R^6, van der Waals 相互作用 (rad/s)
R_b     = (C_6/ℏΩ)^{1/6}, 阻塞半径
F       = Tr(ρ(T) ρ_target), 终态保真度
|W⟩     = (|gr⟩+|rg⟩)/√2, 双原子 Bell 态
|W_3⟩   = (|rgg⟩+|grg⟩+|ggr⟩)/√3, 三原子 W 态（单激发对称叠加）
|g⟩     = 5S_{1/2} 基态
|r⟩     = 53S_{1/2} 里德堡态

单位: 频率用 MHz (显示) / rad·MHz (计算), 长度 μm, 时间 μs
```

---

## Phase 1 — 基础物理 + 代码框架（并行）

### TC-1: Agent-Physics-Text / §1 Rydberg 原子物理基础

**输入**: Spec §6 中 §1 的写作要求 + 参考文献 A 组

**产出**: `drafts/section_01.md`

**内容清单**:
1. §1.1 从氢到碱金属 (~0.5 页)
   - 类氢能级回顾 → 碱金属穿透/极化 → Rydberg-Ritz 公式
   - QDT **简洁版推导**：物理图像（短程相移导致能级偏移），给出结论公式
   - 一句话指向附录 A 完整推导
   - 数值：Rb $\delta_0(nS) \approx 3.13$, $\delta_0(nP) \approx 2.65$, $\delta_0(nD) \approx 1.35$
2. §1.2 标度律 (~0.7 页)
   - Tab.1：标度律总表（7 行：轨道半径、偶极矩、极化率、寿命、BBR 寿命、$C_6$、$R_b$）
   - 每行给出标度指数 + 物理意义一句话
   - 数值锚点：Rb 70S
   - 总结句：长寿命 + 强相互作用 = 量子计算平台基础
3. §1.3 BBR 与寿命修正 (~0.5 页)
   - $1/\tau_{\text{eff}} = 1/\tau_{0K} + \Gamma_{\text{BBR}}$ 公式
   - 物理意义：门时间上限约束
   - 引用 Beterov 2009

**验收标准**:
- [ ] 包含 Rydberg-Ritz 公式（独立编号）
- [ ] 包含 Tab.1 标度律表
- [ ] 包含 BBR 修正公式
- [ ] 所有数值有引用来源
- [ ] QDT 推导简洁（≤0.5 页），明确指向附录 A

---

### TC-2: Agent-Physics-Text / §2 原子—激光相互作用

**输入**: Spec §6 中 §2 的写作要求 + 参考文献 B 组

**产出**: `drafts/section_02.md`

**内容清单**:
1. §2.1 二能级 RWA 推导 (~0.8 页)
   - 偶极近似 → 偶极 Hamiltonian → Rabi 频率定义 → 转动坐标 → RWA
   - 最终有效 Hamiltonian（独立编号公式）
   - Bloch 球图像描述（配合 Fig.4）
   - Rabi 振荡 $P_r(t) = \sin^2(\Omega t/2)$
2. §2.2 双光子激发 (~0.7 页)
   - 实验动机：紫外波长困难
   - 三能级 + 大失谐 → **绝热消除简洁版**
   - 给出 $\Omega_{\text{eff}} = \Omega_1\Omega_2/(2\Delta)$，AC Stark shift，散射代价
   - 指向附录 B 完整推导
   - de Léséleuc 2018 数值锚点
3. §2.3 偏振与选择定则 (~0.3 页)
   - 简短，$m_J$ 通道选择，Walker-Saffman 修正

**验收标准**:
- [ ] RWA Hamiltonian 完整写出
- [ ] 绝热消除给出结论公式，推导简洁（≤0.3 页）
- [ ] 明确指向附录 B
- [ ] 包含数值锚点

---

### TC-3: Agent-Physics-Text / §3 里德堡阻塞与 Bell 态

**输入**: Spec §6 中 §3 的写作要求 + 参考文献 C 组

**产出**: `drafts/section_03.md`

**内容清单**:
1. §3.1 vdW 相互作用 (~0.5 页)
   - 偶极-偶极 → 二阶微扰 → $C_6/R^6$
   - Förster 共振简介
   - $C_6$ 表达式
2. §3.2 阻塞机制**完整推导** (~1.0 页，本报告物理高峰)
   - 4×4 矩阵写出
   - 对称/反对称基变换 $|W\rangle$, $|D\rangle$
   - 暗态脱耦的数学证明
   - 阻塞极限：$V \gg \hbar\Omega$ → 有效二能级
   - $\sqrt{2}$ 增强的物理解释
   - 阻塞半径定义 + Rb 70S 数值
3. §3.3 Bell 态协议 (~0.4 页)
   - $t = \pi/\Omega_{\text{eff}}$ 制备
   - N 原子推广提及
   - 最佳实验数据引用
4. §3.4 实验平台 (~0.3 页)
   - 光镊阵列简述

**验收标准**:
- [ ] 4×4 矩阵完整写出
- [ ] 对称基变换严格推导
- [ ] 阻塞半径公式 + 数值
- [ ] $\sqrt{2}$ 增强有物理解释
- [ ] 这是最详细的推导章节

---

### TC-4: Agent-Simulation / 物理后端代码

**输入**: Spec §5（代码架构）+ §4（实验设置）

**产出**:
- `src/physics/constants.py`
- `src/physics/hamiltonian.py`
- `src/physics/noise_model.py`
- `src/physics/lindblad.py`

**实现要求**:

#### `constants.py`
```python
# 所有参数必须与 Spec §4 和 Tab.2 严格一致
RB87_MASS = 1.4431607e-25   # kg (87 amu)
DELTA_0_S = 3.1311804       # Rb nS quantum defect (Lorenzen & Niemax)
DELTA_0_P = 2.6548849       # Rb nP quantum defect
DELTA_0_D = 1.3480917       # Rb nD quantum defect
TAU_EFF_53S = 88e-6         # s, 含 BBR @ 300K (Beterov 2009)
C6_53S = 2*np.pi * 15.4e9   # rad·Hz·μm^6 (Saffman-Walker-Molmer 2010, Table III 插值)
OMEGA_BASELINE = 2*np.pi * 4.6e6  # rad/s, 场景 B (Evered 2023)
R_ATOM = 2.0e-6             # m, 原子间距
V_VDW_BASELINE = C6_53S / R_ATOM**6  # rad/s
```

#### `hamiltonian.py`
- `build_two_atom_hamiltonian(Omega, Delta, V_vdW)` → 4×4 qutip.Qobj
- `build_three_atom_hamiltonian(Omega, Delta, positions, C6)` → 8×8 qutip.Qobj
- 基矢量顺序明确：$|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle$

#### `noise_model.py`
- `NoiseModel(scenario)` 支持 "A", "B", "D"
- `sample()` 返回字典，键名与公式符号对应
- OU 过程用 Euler-Maruyama 离散化

#### `lindblad.py`
- `get_collapse_operators(n_atoms, tau_eff)` → list[Qobj]
- 封装 `mesolve_with_noise(H, psi0, tlist, c_ops, noise_params)` → Result

**验收标准**:
- [ ] `python -c "from src.physics import *"` 无报错
- [ ] 双原子 Hamiltonian 在 $\Delta=0$, $V=0$ 时的本征值正确
- [ ] 噪声模型在 scenario="B" 时产出所有 5 种噪声
- [ ] Lindblad 演化能复现无噪声下的 Rabi 振荡

---

### TC-5: Agent-Figures / Phase 1 物理图（Fig.1-6）

**输入**: Spec §3（图表清单）+ TC-4 的代码（部分依赖）

**产出**:
- `src/plotting/fig01_energy_levels.py` → `figures/fig01_energy_levels.pdf`
- `src/plotting/fig02_scaling_laws.py` → `figures/fig02_scaling_laws.pdf`
- `src/plotting/fig03_two_photon.py` → `figures/fig03_two_photon.pdf`
- `src/plotting/fig04_rabi_bloch.py` → `figures/fig04_rabi_bloch.pdf`
- `src/plotting/fig05_blockade.py` → `figures/fig05_blockade.pdf`
- `src/plotting/fig06_fidelity_vs_distance.py` → `figures/fig06_fidelity_vs_distance.pdf`

**各图要求**:

**Fig.1 能级图**: Rb 低 $\ell$ 态能级，对比 H 原子虚线，标注 quantum defect 偏移量。用 ARC 数据或手动输入。

**Fig.2 标度律**: 4 子图 log-log：(a) $\langle r \rangle$ vs $n$, (b) $\tau$ vs $n$, (c) $C_6$ vs $n$, (d) $R_b$ vs $n$（固定 $\Omega$）。标注幂律拟合线。

**Fig.3 双光子图**: 三能级梯形图，$5S \to 5P \to nS$，标注 $\Omega_1$, $\Omega_2$, $\Delta$, $\Omega_{\text{eff}}$。简洁明了。

**Fig.4 Rabi 振荡**: (a) 上面板 $P_r(t)$ 共振 + 两条失谐曲线, (b) 下面板或右侧 Bloch 球 3D 轨迹。用 QuTiP Bloch 类。

**Fig.5 阻塞示意**: (a) 双原子能级图：$|gg\rangle, |W\rangle, |rr\rangle$，标注耦合 $\sqrt{2}\Omega$ 和 vdW 位移。(b) 布居演化对比：$V=0$ (自由振荡) vs $V \gg \Omega$ (阻塞振荡)。

**Fig.6 保真度 vs 距离**: $F(|W\rangle)$ 随原子间距 $R$ 变化（固定 $\Omega$, $T=\pi/\sqrt{2}\Omega$），标注 $R_b$ 垂直线。需要 QuTiP mesolve。

**通用绘图规范**:
- 字体：Times New Roman / serif，12pt
- 颜色方案：colorblind-friendly (tableau10 或类似)
- 图尺寸：单栏 3.5 inch，双栏 7 inch
- 所有轴有标签和单位
- 每张图脚本独立可运行
- 输出 PDF（矢量图）

**验收标准**:
- [ ] 所有 6 张图的 Python 脚本存在且可运行
- [ ] 输出 PDF 在 `figures/` 目录
- [ ] 图中的物理参数与 `constants.py` 一致

---

## Phase 2 — 退相干分析 + 基线 + RL 环境（并行）

### TC-6: Agent-Physics-Text / §4 退相干与传统算法失效

**输入**: Spec §6 中 §4 的写作要求 + 参考文献 D 组

**产出**: `drafts/section_04.md`

**内容清单**:
1. §4.1 退相干全景表 (~0.7 页)
   - 表格列出 6 种噪声源（Spec 中已给出完整表格，照搬并中文化）
2. §4.2 线性响应理论 (~0.5 页)
   - Day/PRX Quantum 公式
   - 三条标度律 + 物理解读
   - **双重夹击**论证
3. §4.3 三种传统算法各一段 (~0.8 页)
   - STIRAP: 绝热速度限制 + Goldilocks 区间 + 数值证据
   - GRAPE: 梯度公式（一行） + sim2real gap + barren plateau
   - STA/CD: Berry 公式 + 开系失效
4. 末段总结：自然引出 RL 动机

**验收标准**:
- [ ] 退相干表格完整（6 行）
- [ ] 线性响应公式写出
- [ ] 三种传统算法各有一条关键公式 + 失效机制
- [ ] 末段为 §6 做铺垫

---

### TC-7: Agent-AI-Text / §5-§6

**输入**: Spec §6 中 §5 和 §6 的写作要求 + 参考文献 E 组

**产出**: `drafts/section_05.md`, `drafts/section_06.md`

**§5 内容**:
- 物理系统参数（yaml 格式的文字版）
- 噪声模型 5 种逐项
- Tab.2 场景参数表（A/B/D），标注 B 为主场景
- 评估指标定义（$\langle F \rangle$, $F_{05}$, $\sigma_F$）

**§6 内容**:
- §6.1 MDP 映射（状态 30 维、动作 2 维、转移、奖励）
- §6.2 PPO clipped surrogate objective + 为什么选 PPO（引用 Ernst 2025）
- §6.3 Domain Randomization 物理意义

**验收标准**:
- [ ] §5 参数与 Spec §4 和 `constants.py` 完全一致
- [ ] Tab.2 完整（3 场景 × 6 列）
- [ ] PPO 目标函数独立编号
- [ ] DR 的物理类比（动态退耦）

---

### TC-8: Agent-Simulation / 基线评估

**依赖**: TC-4 完成

**输入**: TC-4 的代码 + Spec §4.3 的场景定义

**产出**:
- `src/baselines/stirap.py`
- `src/baselines/grape.py`
- `src/baselines/evaluate.py`
- `results/baseline_A.json`, `results/baseline_B.json`, `results/baseline_D.json`

**实现要求**:

#### `stirap.py`
- 实现 STIRAP 脉冲：$\Omega_P(t) = \Omega_0 \sin^2(\pi t / 2T)$, $\Omega_S(t) = \Omega_0 \cos^2(\pi t / 2T)$（或等效参数化）
- 可选：反绝热修正项

#### `grape.py`
- 使用 `qutip-qtrl` 的 GRAPE 优化器
- 分段常数脉冲，$N_{\text{steps}} = 30$
- 目标：$|gg\rangle \to |W\rangle$（2-atom）或 $|ggg\rangle \to |W_3\rangle$（3-atom）
- 运行时间：< 10 min on CPU

#### `evaluate.py`
- `evaluate_policy(policy, scenario, n_traj=1000, seed=42)` → dict
- 返回 `{"mean_F", "F_05", "std_F", "all_F"}`
- 支持 STIRAP / GRAPE / PPO 三种 policy 统一接口

#### 结果文件格式
```json
{
  "scenario": "B",
  "method": "GRAPE",
  "n_trajectories": 1000,
  "mean_F": 0.97,
  "F_05": 0.93,
  "std_F": 0.02,
  "params": {...}
}
```

**验收标准**:
- [ ] STIRAP 在场景 A 下无噪声时 $F > 0.99$
- [ ] GRAPE 在场景 B 下无噪声时 $F > 0.99$
- [ ] 加噪声后保真度显著下降
- [ ] 结果 JSON 文件格式一致

---

### TC-9: Agent-RL-Training / PPO 训练

**依赖**: TC-4 完成

**输入**: TC-4 的 physics 代码 + Spec §5（环境设计）

**产出**:
- `src/environments/rydberg_env.py`
- `src/environments/rydberg_env_3atom.py`
- `src/training/config.py`
- `src/training/train_ppo.py`
- `results/ppo_B.json`, `results/ppo_D.json`
- `models/ppo_B_seed{0,1,2}.zip`（SB3 模型文件）

**实现要求**:

#### `rydberg_env.py`
- 继承 `gymnasium.Env`
- observation_space: Box(32,)（$\rho$ 实向量化：将 $4 \times 4$ 复密度矩阵拆为实部和虚部，`np.concatenate([rho.real.flatten(), rho.imag.flatten()])` → 32 维。虽然有冗余，但对 RL agent 更友好。）
- action_space: Box(2,) → $(\Omega(t), \Delta(t))$ 归一化到 $[-1, 1]$
- `reset()`: 重新采样噪声（domain randomization）
- `step()`: 调用 `mesolve` 演化 $dt = T_{\text{gate}} / n_{\text{steps}}$
- 稀疏终态奖励：$r_T = \text{Tr}(\rho(T) \rho_{\text{tgt}})$

#### `config.py`
```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 1.0,           # 无折扣（物理问题，关心终态）
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 500_000,  # 可调，CPU 上 ~几小时
    "n_seeds": 3,
}
```

#### `train_ppo.py`
- 3 个 seed 分别训练
- 训练完后自动调用 `evaluate.py` 评估
- 保存学习曲线日志（CSV 或 TensorBoard）

**验收标准**:
- [ ] 环境 `check_env()` 通过
- [ ] PPO 在场景 B（无噪声）下训练后 $F > 0.95$
- [ ] PPO 在场景 B（全噪声）下 $\langle F \rangle >$ GRAPE 的 $\langle F \rangle$
- [ ] 3 个 seed 的训练曲线存在

---

## Phase 3 — 结果可视化 + 结果写作（并行）

### TC-10: Agent-Figures / Phase 3 结果图（Fig.7-14）

**依赖**: TC-8 + TC-9 完成

**输入**: `results/*.json` + 训练日志

**产出**:
- `src/plotting/fig07_noise_impact.py` → `figures/fig07_noise_impact.pdf`
- `src/plotting/fig08_traditional_limits.py` → `figures/fig08_traditional_limits.pdf`
- `src/plotting/fig09_mdp_schematic.py` → `figures/fig09_mdp_schematic.pdf`
- `src/plotting/fig10_main_comparison.py` → `figures/fig10_main_comparison.pdf`
- `src/plotting/fig11_robustness.py` → `figures/fig11_robustness.pdf`
- `src/plotting/fig12_training_curve.py` → `figures/fig12_training_curve.pdf`
- `src/plotting/fig13_pulse_shapes.py` → `figures/fig13_pulse_shapes.pdf`
- `src/plotting/fig14_population_evolution.py` → `figures/fig14_population_evolution.pdf`

**各图详细要求**:

**Fig.7 噪声影响**: 分组柱状图，X 轴 = 噪声源（Doppler, 位置, 幅度, 相位, 衰减, 全部），Y 轴 = $1 - \langle F \rangle$。场景 B 参数。

**Fig.8 传统算法极限**: $\langle F \rangle$ vs $T_{\text{gate}}$，三条曲线（STIRAP, GRAPE, STA），标注各自的天花板。

**Fig.9 MDP 示意**: 流程图风格，状态 $\rho(t)$ → Agent → $(\Omega, \Delta)$ → Lindblad 演化 → $\rho(t+dt)$ → ... → $F$。

**Fig.10 核心对比（最重要的图）**: 场景 B 下 STIRAP/GRAPE/PPO 三种方法的 $\langle F \rangle$ 柱状图。带误差棒（$\sigma_F$）。如果有余力，加场景 A 和 D 作为子图。

**Fig.11 鲁棒性**: $\langle F \rangle$ vs $\delta\Omega/\Omega \in [0, 5\%]$，三条线。展示 PPO 在噪声增大时保真度下降更慢。

**Fig.12 学习曲线**: episode reward vs training step，3 seeds 分别画细线 + 均值粗线 + shaded std。

**Fig.13 脉冲形状**: 双面板。(a) $\Omega(t)$ 对比 (b) $\Delta(t)$ 对比。PPO（蓝色）vs GRAPE（红色）。

**Fig.14 态演化**: $P_{|gg\rangle}(t)$, $P_{|W\rangle}(t)$, $P_{|rr\rangle}(t)$ 三条线，对比 PPO vs GRAPE（实线 vs 虚线）。

**验收标准**:
- [ ] 8 张图全部生成 PDF
- [ ] Fig.10 是报告核心图，必须清晰美观
- [ ] 所有数据来自实际运行结果（`results/*.json`）

---

### TC-11: Agent-AI-Text / §7-§8

**依赖**: TC-8 + TC-9 + TC-10 完成

**输入**: `results/*.json` + 所有图表 + Spec §6 写作要求

**产出**: `drafts/section_07.md`, `drafts/section_08.md`

**§7 内容**:
- §7.1 实现说明：Plan B 技术栈、代码结构简述
- §7.2 核心结果（基于场景 B）：
  - 引用 Fig.10 柱状图
  - 报告 $\langle F \rangle$, $F_{05}$, $\sigma_F$ 的具体数值
  - 解读：PPO 在哪些方面超过 GRAPE
- §7.3 鲁棒性分析（引用 Fig.11）
- §7.4 训练细节（引用 Fig.12, Fig.13）
- §7.5 辅助结果（场景 A, D 各一小段）
- §7.6 诚实可行性声明

**§8 内容**:
- 物理证明链总结
- AI 的定位
- 局限性（$4^N$ 维灾难、sim-to-real）
- 展望

**验收标准**:
- [ ] 所有数值来自 `results/*.json`，不是编造的
- [ ] 所有"如图 X 所示"的交叉引用对应正确的图
- [ ] 包含诚实的可行性声明

---

### TC-12: Agent-Physics-Text / 附录 A-E

**输入**: Spec 附录要求 + 参考文献

**产出**: `drafts/appendix_A.md` 至 `drafts/appendix_E.md`

**各附录要求**:

**附录 A: QDT 完整推导 (3-4 页)**
- WKB 近似 → 径向波函数在 $r \to \infty$ 的渐近形式
- 短程相移：非库仑势区域的散射相位
- quantum defect 与相移的关系：$\delta_{\ell} = \pi \mu_\ell$
- Rydberg-Ritz 公式的严格导出
- 修正项 $\delta_2/(n-\delta_0)^2 + \cdots$ 的物理来源

**附录 B: 绝热消除 (2-3 页)**
- 三能级 Hamiltonian 完整写出
- 投影算符方法 / Schrieffer-Wolff 变换
- 逐步推导有效 Hamiltonian
- AC Stark shift 和散射率的导出

**附录 C: $C_6$ 通道求和 (2 页)**
- 偶极-偶极算符在 $|n\ell m\rangle$ 基下的矩阵元
- Zeeman 简并下的求和
- Förster 缺陷与共振条件

**附录 D: GRAPE 梯度 + Barren Plateau (2 页)**
- GRAPE 梯度公式完整推导
- Larocca 类型 barren plateau 的直觉解释

**附录 E: Lindblad 数值方法 (1-2 页)**
- 主方程 → Kraus 表示
- mesolve vs mcsolve 对比
- 数值积分注意事项

**验收标准**:
- [ ] 每个附录自含，可独立阅读
- [ ] 数学推导步骤完整，没有跳步
- [ ] 与正文的简洁版一致（结论相同）

---

### TC-13: Agent-AI-Text / 附录 F

**输入**: PPO 算法 + 训练超参数

**产出**: `drafts/appendix_F.md`

**内容**:
- PPO 完整伪代码（Algorithm 环境格式）
- 超参数表（与 `config.py` 一致）
- 环境详细参数

**验收标准**:
- [ ] 伪代码可读
- [ ] 超参数与实际训练一致

---

## Phase 4 — 整合

### TC-14: Main Agent / 组装与审查

**依赖**: 所有 TC 完成

**输入**: `drafts/section_*.md` + `drafts/appendix_*.md` + `figures/*.pdf` + `results/*.json`

**产出**: `report.md`

**任务**:
1. 按顺序组装所有 section 和附录
2. 添加 Abstract（基于全文内容撰写）
3. 添加参考文献（从 Spec §8 复制，确保所有引用都在正文中出现过）
4. **一致性审查**:
   - 所有物理参数值在正文、代码、图表中一致
   - 所有"如图 X 所示"交叉引用正确
   - 所有公式编号连续
   - 符号使用统一（$\Omega$ 不混用 $\omega$）
5. **逻辑链审查**:
   - §1-3 的物理基础是否充分支撑 §4 的失效论证
   - §4 的失效是否自然引出 §6 的 RL 方案
   - §7 的结果是否回答了 §5 提出的问题

**验收标准**:
- [ ] `report.md` 编译无 LaTeX 错误（在 markdown 预览中公式渲染正确）
- [ ] 所有 14 张图的引用都存在
- [ ] 参考文献完整
- [ ] 无残留的 TODO / TBD / placeholder

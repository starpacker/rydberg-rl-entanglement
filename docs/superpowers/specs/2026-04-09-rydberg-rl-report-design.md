# Design Spec: 里德堡原子 RL 控制期中报告

> **日期**: 2026-04-09
> **状态**: Draft
> **目标**: 一份完全可执行的 spec，可直接交给 main-agent 指挥 sub-agent 集群完成全部写作 + 代码 + 出图

---

## 1. 项目元数据

| 属性 | 值 |
|---|---|
| **报告标题** | 驯服量子混沌：里德堡原子阵列中基于强化学习的高保真度纠缠态制备 |
| **英文标题** | Taming Quantum Chaos: High-Fidelity Entanglement Preparation in Rydberg Atom Arrays via Reinforcement Learning |
| **课程** | 原子物理学期中报告 |
| **语言** | 中文正文 + 英文术语（如 Rydberg blockade、PPO、domain randomization） |
| **风格** | 前沿专题型；物理为主体（≥70%），AI 作为应用收尾（≤30%） |
| **数学深度** | 核心结论在正文给出简洁推导，附录补充完整版 |
| **时间约束** | 1 天内完成 |

---

## 2. 报告结构（章节 + 预估篇幅）

### 正文

| Part | 标题 | 核心内容 | 关键图表 |
|---|---|---|---|
| **Abstract** | 摘要 | 问题—方法—结果一句话 | — |
| **§1** | Rydberg 原子的物理基础 | QDT（正文简洁版）+ 标度律 + BBR 寿命 | Fig.1 能级图, Fig.2 标度律, Tab.1 标度表 |
| **§2** | 原子—激光相互作用 | RWA 推导 + 双光子激发 + 绝热消除（正文简洁版） | Fig.3 双光子能级图, Fig.4 Rabi 振荡/Bloch 球 |
| **§3** | 里德堡阻塞与 Bell 态制备 | vdW 推导 + 阻塞机制 + Bell 态协议 | Fig.5 阻塞机制示意, Fig.6 保真度 vs 距离 |
| **§4** | 退相干预算与传统算法失效 | 全景退相干表 + 线性响应 + STIRAP/GRAPE/STA 失效 | Fig.7 噪声源全景图, Fig.8 传统算法保真度上限 |
| **§5** | 实验设置 | 场景 A/B/D 定义（B 为主、A/D 辅助） | Tab.2 场景参数表 |
| **§6** | AI 控制：PPO + Domain Randomization | MDP 映射 + PPO 核心 + DR 物理意义 | Fig.9 MDP 示意图 |
| **§7** | 复现与结果 | Plan B 实现 + 核心对比结果 | Fig.10-14（见图表清单） |
| **§8** | 讨论与结论 | 证明链总结 + 局限 + 展望 | — |

### 附录

| 附录 | 内容 |
|---|---|
| **A** | Quantum Defect Theory 完整推导（WKB 相位积分 + 短程相移） |
| **B** | 双光子绝热消除完整推导（投影算符 / Schrieffer-Wolff） |
| **C** | Walker-Saffman 含 Zeeman 简并的 $C_6$ 通道求和 |
| **D** | GRAPE 梯度完整推导 + Barren Plateau 简证 |
| **E** | Lindblad 主方程 Kraus 表示与数值积分细节 |
| **F** | PPO 完整伪代码 + 超参数表 |

---

## 3. 图表清单（14 张图 + 2 张表）

### 物理示意图（由绘图代码生成，matplotlib/tikz）

| ID | 图名 | 所属章节 | 内容描述 | 生成方式 |
|---|---|---|---|---|
| **Fig.1** | Rb 能级图 | §1.1 | $^{87}$Rb 低角动量态能级图，标注 quantum defect 偏移，对比氢原子 | matplotlib + ARC 数据 |
| **Fig.2** | 标度律可视化 | §1.2 | 关键物理量（$\langle r \rangle$, $C_6$, $\tau$, $R_b$）随 $n^*$ 的 log-log 图 | matplotlib + ARC 计算 |
| **Fig.3** | 双光子激发能级图 | §2.2 | $5S_{1/2} \to 5P_{3/2} \to nS/nD$ 梯形图，标注 $\Omega_1$, $\Omega_2$, $\Delta$ | matplotlib 手绘风格 |
| **Fig.4** | Rabi 振荡 + Bloch 球 | §2.1 | (a) $P_r(t)$ 共振/失谐曲线 (b) Bloch 球轨迹 | matplotlib + QuTiP Bloch |
| **Fig.5** | 阻塞机制示意 | §3.2 | (a) 双原子能级图含 $V_{\text{vdW}}$ 位移 (b) 阻塞 vs 非阻塞动力学对比 | matplotlib |
| **Fig.6** | 保真度 vs 原子间距 | §3.2 | $F(|W\rangle)$ 随 $R$ 变化，标注 $R_b$ 位置 | QuTiP mesolve 计算 |

### 动力学演化图（需 QuTiP 模拟）

| ID | 图名 | 所属章节 | 内容描述 | 生成方式 |
|---|---|---|---|---|
| **Fig.7** | 噪声源影响对比 | §4.1 | 各噪声源单独/叠加对保真度的影响（分组柱状图） | QuTiP Monte Carlo |
| **Fig.8** | 传统算法保真度上限 | §4.3 | STIRAP/GRAPE/STA 在场景 B 下的保真度 vs 门时间曲线 | QuTiP + qutip-qtrl |

### 算法对比 & RL 结果图（需训练 + 模拟）

| ID | 图名 | 所属章节 | 内容描述 | 生成方式 |
|---|---|---|---|---|
| **Fig.9** | MDP 映射示意 | §6.1 | 状态/动作/奖励的流程图 | matplotlib 手绘 |
| **Fig.10** | **核心结果：场景 B 算法对比** | §7 | STIRAP/GRAPE/PPO 在场景 B 下的 $\langle F \rangle$ 柱状图 + 误差棒 | 训练结果 |
| **Fig.11** | 鲁棒性曲线 | §7 | $\langle F \rangle$ vs $\delta\Omega/\Omega \in [0, 5\%]$，三种方法对比 | 参数扫描 |
| **Fig.12** | PPO 训练学习曲线 | §7 | reward vs episodes，3 seeds，含 shaded std | SB3 训练日志 |
| **Fig.13** | 脉冲形状对比 | §7 | PPO 策略的 $(\Omega(t), \Delta(t))$ vs GRAPE 最优脉冲 | 训练结果提取 |
| **Fig.14** | 态演化对比 | §7 | 场景 B 下 PPO vs GRAPE 的布居数 $P_{|gg\rangle}, P_{|W\rangle}, P_{|rr\rangle}$ 随时间演化 | QuTiP mesolve |

### 表格

| ID | 表名 | 所属章节 | 内容 |
|---|---|---|---|
| **Tab.1** | 标度律总表 | §1.2 | 物理量、标度指数、物理意义、Rb 70S 数值 |
| **Tab.2** | 实验场景参数表 | §5 | 场景 A/B/D 的 $T_{\text{gate}}$, $\Omega$, 噪声配置, 预期传统结果 |

---

## 4. 实验设置详细定义

### 4.1 物理系统（所有场景共享）

```yaml
atom: 87Rb
rydberg_state: "53S_{1/2}, m_J=+1/2"
tau_eff: 88  # μs, 含 BBR @ 300K
excitation: "two-photon via 6P_{3/2}"
intermediate_detuning: 7.8  # GHz
target_state: "|W⟩ = (|gr⟩+|rg⟩)/√2"
```

### 4.2 噪声模型

```yaml
noise_model:
  doppler:
    type: "gaussian_static"
    sigma: 50  # kHz (T=10 μK)
  position_jitter:
    type: "gaussian_static"
    sigma_R: 100  # nm
    effect: "V_vdW(R+δR) via 1/R^6"
  laser_amplitude:
    type: "ornstein_uhlenbeck"
    correlation_time: 10  # μs
    sigma: 0.02  # relative
  phase_servo_bump:
    type: "PSD_peak"
    frequency: "Ω/2π"
    height: -80  # dBc/Hz
  rydberg_decay:
    type: "lindblad"
    operator: "sqrt(1/tau_eff) |g⟩⟨r|"
```

### 4.3 三组对照场景

| 场景 | 描述 | $R$ (μm) | $T_{\text{gate}}$ | $\Omega/2\pi$ | 噪声 | 角色 |
|---|---|---|---|---|---|---|
| **A** | 长时间 / STIRAP 兼容 | 2 | 5 μs | 0.8 MHz | 仅 doppler + decay | 辅助：展示绝热极限 |
| **B** | 短时间 / 全噪声（**主场景**） | 2 | 0.3 μs | 4.6 MHz | 全部 5 种 | **核心**：传统 vs RL 对决 |
| **D** | 3 原子 W 态制备 | 2 (等边三角) | 0.5 μs | 4.6 MHz | 全部 5 种 | 辅助：展示可扩展性 |

### 4.4 评估指标

```yaml
evaluation:
  metric: "Tr(ρ(T) ρ_target)"
  target_B: "|W⟩⟨W|"  # 2-atom
  target_D: "|W_3⟩⟨W_3|"  # 3-atom W state
  monte_carlo_trajectories: 1000
  reported_stats:
    - mean_F: "⟨F⟩ over 1000 trajectories"
    - worst_5pct: "F_{05}"
    - std_F: "σ_F"
```

---

## 5. 代码架构（Plan B 轻量栈）

### 5.1 技术栈

```yaml
dependencies:
  physics:
    - qutip >= 5.0
    - numpy
    - scipy
  rl:
    - gymnasium
    - stable-baselines3
  baseline_control:
    - qutip-qtrl  # GRAPE / Krotov
  visualization:
    - matplotlib
    - seaborn
  optional:
    - ARC-Alkali-Rydberg-Calculator  # 能级数据
```

### 5.2 目录结构

```
report/
├── report.md                    # 最终报告 markdown
├── figures/                     # 所有生成的图表
│   ├── fig01_energy_levels.pdf
│   ├── fig02_scaling_laws.pdf
│   ├── ...
│   └── fig14_population_evolution.pdf
├── src/
│   ├── physics/
│   │   ├── hamiltonian.py       # Rydberg Hamiltonian 构建（2-atom, 3-atom）
│   │   ├── noise_model.py       # 噪声采样器（Doppler, OU, 位置抖动等）
│   │   ├── lindblad.py          # Lindblad 算符 + mesolve 封装
│   │   └── constants.py         # 物理常数 + Rb 参数
│   ├── environments/
│   │   ├── rydberg_env.py       # Gymnasium 环境（2-atom Bell 态）
│   │   └── rydberg_env_3atom.py # Gymnasium 环境（3-atom W 态）
│   ├── baselines/
│   │   ├── stirap.py            # STIRAP 脉冲生成 + 模拟
│   │   ├── grape.py             # GRAPE 优化（via qutip-qtrl）
│   │   └── evaluate.py          # 统一评估接口（1000 MC trajectories）
│   ├── training/
│   │   ├── train_ppo.py         # PPO 训练脚本
│   │   └── config.py            # 超参数配置
│   └── plotting/
│       ├── fig_energy_levels.py
│       ├── fig_scaling_laws.py
│       ├── fig_two_photon.py
│       ├── fig_rabi_bloch.py
│       ├── fig_blockade.py
│       ├── fig_fidelity_vs_distance.py
│       ├── fig_noise_impact.py
│       ├── fig_traditional_limits.py
│       ├── fig_mdp_schematic.py
│       ├── fig_main_comparison.py
│       ├── fig_robustness.py
│       ├── fig_training_curve.py
│       ├── fig_pulse_shapes.py
│       └── fig_population_evolution.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-09-rydberg-rl-report-design.md  # 本文件
└── requirements.txt
```

### 5.3 关键接口定义

#### `physics/hamiltonian.py`

```python
def build_two_atom_hamiltonian(
    Omega: float,          # Rabi 频率 (rad/s)
    Delta: float,          # 失谐 (rad/s)
    V_vdW: float,          # vdW 相互作用 (rad/s)
    basis: str = "symmetric"  # "full" | "symmetric"
) -> qutip.Qobj:
    """返回双原子 Rydberg Hamiltonian (4x4 或 3x3 对称子空间)"""

def build_three_atom_hamiltonian(
    Omega: float,
    Delta: float,
    positions: np.ndarray,  # shape (3, 2), μm
    C6: float,              # vdW 系数
) -> qutip.Qobj:
    """返回 3 原子 Hamiltonian (8x8)"""
```

#### `physics/noise_model.py`

```python
class NoiseModel:
    def __init__(self, scenario: str):  # "A", "B", or "D"
        """根据场景加载噪声参数"""

    def sample(self, rng: np.random.Generator) -> dict:
        """采样一组噪声实现，返回 {delta_doppler, delta_R, xi_amplitude, ...}"""

    def get_lindblad_ops(self) -> list[qutip.Qobj]:
        """返回 Lindblad collapse operators"""
```

#### `environments/rydberg_env.py`

```python
class RydbergBellEnv(gymnasium.Env):
    """
    Gymnasium 环境：2-atom Bell 态制备
    - observation: ρ(t) 的实向量化 (32 维, real+imag flatten)
    - action: (Omega(t), Delta(t)) ∈ [-1, 1]^2 (归一化后映射到物理范围)
    - reward: 稀疏终态 Tr(ρ(T) ρ_target)
    - step(): 调用 QuTiP mesolve 演化一个时间片 dt
    """
    def __init__(self, scenario: str = "B", n_steps: int = 30):
        ...
    def reset(self, seed=None):
        """重新采样噪声参数 (domain randomization)"""
    def step(self, action):
        """演化 dt，返回 obs, reward, terminated, truncated, info"""
```

#### `baselines/evaluate.py`

```python
def evaluate_policy(
    policy: Callable,       # action = policy(obs, t)
    scenario: str,          # "A", "B", "D"
    n_trajectories: int = 1000,
    seed: int = 42,
) -> dict:
    """
    返回 {"mean_F": float, "F_05": float, "std_F": float, "all_F": np.array}
    """
```

---

## 6. 各章节写作要求

### §1 Rydberg 原子物理基础

**QDT 正文版（简洁）**：
- 从氢原子出发，说明碱金属核外电子的穿透/极化效应
- 给出 Rydberg-Ritz 公式 $E_{n\ell j} = -\text{Ry}^*/(n-\delta_{n\ell j})^2$
- 物理解释 quantum defect 的含义（短程相移）
- **不展开** WKB 积分细节，用一句话指向附录 A
- 数值示例：Rb 的 $\delta_0(nS) \approx 3.13$

**标度律**：
- 必须包含 Tab.1（完整标度表）
- 每个标度律给出一句话物理解释
- 数值锚点：Rb 70S 的具体数值
- 结论句：里德堡态兼具长寿命和强相互作用

**BBR 寿命修正**：
- 公式：$1/\tau_{\text{eff}} = 1/\tau_{0K} + \Gamma_{\text{BBR}}$
- 物理意义：$T_{\text{gate}} \ll \tau_{\text{eff}}$ 的约束

### §2 原子—激光相互作用

**RWA 推导**：
- 完整但不冗长：偶极近似 → 偶极 Hamiltonian → 转动坐标 → RWA
- 必须写出最终有效 Hamiltonian
- Bloch 球图像 + Rabi 振荡公式

**绝热消除正文版（简洁）**：
- 三能级 $\Lambda$ 系统 → 大失谐 → 中间态绝热消除
- 给出有效 Rabi 频率 $\Omega_{\text{eff}} = \Omega_1\Omega_2/(2\Delta)$
- 散射代价公式
- **不展开** Schrieffer-Wolff 变换细节，指向附录 B
- 数值锚点：de Léséleuc 2018

### §3 里德堡阻塞与 Bell 态

**这是物理顶峰章节，需要最充分的推导**：
- vdW 从 $1/R^3$ 到 $1/R^6$ 的物理图像
- **完整的** 4×4 矩阵 → 对称基变换 → 阻塞极限化简
- $\sqrt{2}$ Rabi 增强的物理解释
- 阻塞半径定义 + 数值
- Bell 态协议：$t = \pi/\Omega_{\text{eff}}$ 时 $|gg\rangle \to -i|W\rangle$

### §4 退相干与传统算法失效

**退相干全景**：
- 用表格列出所有噪声源 + 数学描述 + 典型值
- 线性响应公式 $1-F \approx \sum_j \int S_j(f) I_j(f) df$

**传统算法失效（每种一段话 + 一条关键公式）**：
- STIRAP：绝热速度限制 + 被衰减锁死
- GRAPE：Sim2Real gap + barren plateau
- STA/CD：开系下暗态失效
- **末段总结**：为什么需要无模型 RL

### §5 实验设置

- **简洁但精确**：给出 Tab.2，每个场景的参数完整、可复现
- **重点强调场景 B**：解释为什么它最能展示 RL 优势
- A 和 D 各一小段说明其补充角色

### §6 AI 控制

- MDP 映射：状态/动作/转移/奖励的精确定义
- PPO：写出 clipped surrogate objective，解释为什么选 PPO
- Domain Randomization：物理意义 = 学习对噪声鲁棒的反馈策略

### §7 复现与结果

- Plan B 实现路线说明
- **核心结果**：场景 B 下 STIRAP/GRAPE/PPO 的 $\langle F \rangle$ 对比
- 鲁棒性曲线
- 训练曲线 + 脉冲形状对比
- **诚实声明**：不追求 0.9996，目标是定性证明 PPO + DR 的优势

### §8 讨论与结论

- 物理证明链回顾
- AI 的定位：不是魔法，是无模型鲁棒控制
- 局限：2-atom 到多 atom 的 $4^N$ 困难；sim-to-real gap
- 展望：Sr/Yb 碱土类 + 可微量子模拟器

### 附录

- 每个附录必须自含，读者可独立阅读
- 附录 A：QDT 完整推导（WKB + 相移），3-4 页
- 附录 B：绝热消除完整推导（Schrieffer-Wolff），2-3 页
- 附录 C：$C_6$ 通道求和，2 页
- 附录 D：GRAPE 梯度 + barren plateau，2 页
- 附录 E：Lindblad 数值方法，1-2 页
- 附录 F：PPO 伪代码 + 超参数表，1 页

---

## 7. Sub-Agent 分工

### Agent 架构

```
Main Agent (Orchestrator)
├── Agent-Physics-Text     [§1-§4, §8, 附录 A-E]
├── Agent-AI-Text          [§5-§7, 附录 F]
├── Agent-Simulation       [物理模拟代码 + 基线评估]
├── Agent-RL-Training      [环境 + PPO 训练]
└── Agent-Figures          [所有 14 张图]
```

### 依赖图

```
Phase 1 (并行):
  Agent-Physics-Text: §1-§3 正文
  Agent-Simulation:   hamiltonian.py, noise_model.py, lindblad.py, constants.py
  Agent-Figures:      Fig.1-6（纯物理图，不依赖训练结果）

Phase 2 (Phase 1 完成后，并行):
  Agent-Physics-Text: §4 + 附录 A-E
  Agent-AI-Text:      §5-§6
  Agent-Simulation:   baselines/ (STIRAP + GRAPE 评估)
  Agent-RL-Training:  rydberg_env.py + train_ppo.py（依赖 Phase 1 的 physics/）

Phase 3 (Phase 2 完成后，并行):
  Agent-AI-Text:      §7（依赖训练结果）+ §8 + 附录 F
  Agent-Figures:      Fig.7-14（依赖模拟和训练结果）

Phase 4 (汇总):
  Main Agent:         组装 report.md + 一致性审查 + 交叉引用检查
```

### 各 Agent 产出物

| Agent | 产出 | 格式 |
|---|---|---|
| Agent-Physics-Text | §1-§4, §8 正文 + 附录 A-E | markdown (LaTeX math) |
| Agent-AI-Text | §5-§7 正文 + 附录 F | markdown (LaTeX math) |
| Agent-Simulation | `src/physics/`, `src/baselines/`, 评估结果 JSON | Python + JSON |
| Agent-RL-Training | `src/environments/`, `src/training/`, 训练日志 + 模型 | Python + tensorboard logs |
| Agent-Figures | `figures/fig01-fig14.pdf` + 绘图脚本 | Python + PDF |

---

## 8. 参考文献（按主题分组，arXiv 编号已核对）

### A. 里德堡物理基础
- [SWM10] Saffman, Walker, Molmer, *Rev. Mod. Phys.* **82**, 2313 (2010). arXiv:0909.4777
- [Gallagher94] Gallagher, *Rydberg Atoms*, Cambridge (1994)
- [SA18] Sibalic & Adams, *Rydberg Physics*, IOP (2018)
- [Beterov09] Beterov et al., *PRA* **79**, 052504 (2009). arXiv:0810.0339

### B. 原子—激光相互作用
- [dL18] de Leseluc et al., *PRA* **97**, 053803 (2018). arXiv:1802.10424

### C. 阻塞与实验
- [Lukin01] Lukin et al., *PRL* **87**, 037901 (2001). arXiv:quant-ph/0011028
- [Urban09] Urban et al., *Nat. Phys.* **5**, 110 (2009). arXiv:0805.0758
- [Gaetan09] Gaetan et al., *Nat. Phys.* **5**, 115 (2009). arXiv:0810.2960
- [WS08] Walker & Saffman, *PRA* **77**, 032723 (2008). arXiv:0712.3438
- [Levine19] Levine et al., *PRL* **123**, 170503 (2019). arXiv:1908.06101
- [Evered23] Evered et al., *Nature* **622**, 268 (2023). arXiv:2304.05420
- [Scholl23] Scholl et al., *Nature* **622**, 273 (2023). arXiv:2305.03406
- [BL20] Browaeys & Lahaye, *Nat. Phys.* **16**, 132 (2020). arXiv:2002.07413

### D. 传统量子控制
- [Vitanov17] Vitanov et al., *Rev. Mod. Phys.* **89**, 015006 (2017). arXiv:1605.00224
- [Khaneja05] Khaneja et al., *J. Magn. Reson.* **172**, 296 (2005)
- [Goerz14] Goerz et al., *PRA* **90**, 032329 (2014). arXiv:1401.1858
- [Larocca22] Larocca et al., *Quantum* **6**, 824 (2022). arXiv:2105.14377
- [Day22] Day, Ramette, Schleier-Smith, *npj QI* **8**, 72 (2022). arXiv:2112.04946
- [PRXQ25] *PRX Quantum* **6**, 010331 (2025). arXiv:2407.20184
- [YB23] Yague Bosch et al., *Ann. Phys.* (2023). arXiv:2312.11594

### E. RL & AI 控制
- [Bukov18] Bukov et al., *PRX* **8**, 031086 (2018). arXiv:1705.00565
- [Niu19] Niu et al., *npj QI* **5**, 33 (2019). arXiv:1803.01857
- [Ernst25] Ernst et al., ICML 2025. arXiv:2501.14372
- [DingEnglund25] Ding & Englund, arXiv:2504.11737 (2025)
- [Guatto24] Guatto et al., *Nat. Commun.* **15**, 8901 (2024)
- [PPO17] Schulman et al., arXiv:1707.06347

### F. 代码/工具
- [krotov] https://github.com/qucontrol/krotov
- [qutip-qtrl] https://github.com/qutip/qutip-qtrl
- [ARC] https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator
- [SB3] https://github.com/DLR-RM/stable-baselines3

---

## 9. 质量标准

### 写作标准
- 中文正文，术语保留英文
- 所有公式用 LaTeX 渲染，重要公式独占一行并编号
- 每个章节开头有 1-2 句导言，说明该章节的目标和与上下文的衔接
- 引用格式：[Author YY] 行内引用
- 避免空泛描述，每个论断都有公式或数据支撑

### 代码标准
- 类型标注（type hints）
- docstring（Google style）
- 所有物理常数集中在 `constants.py`
- 随机数生成器统一通过 `np.random.Generator` 传入（可复现）
- 绘图脚本独立可运行：`python src/plotting/fig_xxx.py` → `figures/fig_xxx.pdf`

### 一致性要求
- 正文中的参数值必须与 `constants.py` 和 Tab.2 完全一致
- 图表中的数据必须来自实际运行的代码，不允许手造数据
- 交叉引用：正文提到"如图 X 所示"时，图 X 必须存在且内容匹配

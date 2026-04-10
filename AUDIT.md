# 项目诚信审计报告 (Integrity Audit)

> **审计日期**: 2026-04-10
> **审计范围**: 报告全文 (`report.md`)、全部代码 (`src/`)、全部结果 (`results/`)、全部图表 (`figures/`)
> **审计原则**: 对每个数据点追溯到生成它的代码和计算过程。无法追溯的一律标记。

---

## 审计结论概要

| 类别 | 真实计算 | 合成/伪造 | 未验证声明 |
|------|---------|----------|-----------|
| 核心数值结果 (Table 3, 4) | **是** — STIRAP/GRAPE/PPO 均有真实 Monte Carlo 评估 | — | 表中数值经过四舍五入，存在微小偏差 |
| PPO 训练过程 | **是** — 3 seed × 1706 episode 逐条记录，模型文件存在 | — | — |
| 物理模拟内核 | **是** — Hamiltonian、Lindblad、噪声模型物理正确 | — | RL 环境噪声模型有简化（见下） |
| 图表 (14 幅) | 9 幅真实 | **5 幅含合成数据** | — |
| 场景 D (3-atom) | — | — | **从未执行任何计算** |
| 报告文字声明 | — | — | **多处数值错误或夸大** |

---

## 一、真实运行并可验证的部分

### 1.1 PPO 训练 — 真实

**证据链**:
- `src/training/train_ppo.py` 使用 SB3 PPO + `RydbergBellEnv` Gymnasium 环境
- `results/training_logs.json` 大小 ~110k tokens，包含 3 个种子（42, 153, 264）各 1706 episode 的逐 episode 保真度
- 训练曲线从 F ≈ 0.04 起步（随机策略），逐步上升，符合真实学习动态
- `models/` 目录下有 4 个 `.zip` 文件：`ppo_B_seed42.zip`, `ppo_B_seed153.zip`, `ppo_B_seed264.zip`, `ppo_B_best.zip`
- wall_time: seed 42 = 183s, seed 153 = 109s, seed 264 = 111s, 总计 405s

**但存在的问题**: 见 §三.1（RL 环境噪声简化）

### 1.2 STIRAP / GRAPE 基线评估 — 真实

**证据链**:
- `src/baselines/evaluate.py` → `evaluate_policy()` 循环 N 次调用 `NoiseModel.sample()` 采样噪声，逐轨迹调用 `run_stirap()` / `run_grape_eval()`
- `run_stirap()` 调用 `qutip.mesolve()` 求解完整 Lindblad 主方程
- `run_grape_eval()` 同样通过 `_evaluate_with_noise()` 调用 `qutip.mesolve()`
- 结果文件浮点精度（如 `mean_F: 0.9963788655590855`）符合真实计算特征

**结果文件**:
| 文件 | 来源 | 轨迹数 |
|------|------|--------|
| `stirap_A.json` | `run_stirap("A")` × 100 MC | 100 |
| `stirap_B.json` | `run_stirap("B")` × 100 MC | 100 |
| `grape_B.json` | GRAPE 无噪声优化 → 有噪声评估 × 100 MC | 100 |
| `ppo_B.json` | 最优 seed 模型 × 200 deterministic eval | 200 |

### 1.3 物理模拟内核 — 正确

| 模块 | 审计结论 |
|------|---------|
| `constants.py` | 基本物理常数正确 (CODATA 2018)；Rb-87 量子亏损符合 Li et al.；C6_53S = 2π×15.4 GHz·μm⁶ 量级正确但难以独立精确验证 |
| `hamiltonian.py` | 旋转框架下的 Rydberg Hamiltonian 构建正确；两原子和三原子情况均实现 |
| `noise_model.py` | 5 个噪声通道实现正确；OU 过程使用精确离散化公式 |
| `lindblad.py` | 坍缩算符 `L = sqrt(γ)|g⟩⟨r|` 正确；调用 QuTiP `mesolve` 求解 |

### 1.4 图表 — 基于真实数据的 (9/14)

| 图 | 数据来源 | 状态 |
|----|---------|------|
| Fig.01 能级图 | Rydberg-Ritz 解析公式 + `constants.py` 量子亏损 | **合法** (解析物理) |
| Fig.02 标度律 | 解析幂律 n*² / n*³ / n*¹¹ | **合法** (教科书公式) |
| Fig.03 双光子 | 示意图，无数据 | **合法** (diagram) |
| Fig.04 Rabi/Bloch | `qutip.mesolve` 实时计算 | **合法** (真实模拟) |
| Fig.05 Blockade | `qutip.mesolve` 实时计算 | **合法** (真实模拟) |
| Fig.06 F vs R | 80 点参数扫描，`qutip.mesolve` | **合法** (真实模拟) |
| Fig.09 MDP 示意 | 示意图，无数据 | **合法** (diagram) |
| Fig.10 场景 B 对比 | 加载 `results/*.json` | **合法** (加载真实结果) |
| Fig.12 训练曲线 | 加载 `training_logs.json` | **合法** (加载真实结果) |

---

## 二、合成/伪造数据的部分

### 2.1 Fig.07 — 噪声影响柱状图 — **完全伪造**

**文件**: `src/plotting/fig07_noise_impact.py`
**代码注释**: `"Synthetic infidelity data (literature-calibrated)"`
**问题**: 第 24 行硬编码了 6 个 infidelity 值：
```python
infidelities = [0.003, 0.001, 0.002, 0.001, 0.0005, 0.004]
```
这些数字不来自任何计算。没有运行"逐通道开启单个噪声源"的扫描实验。

**如何真实验证**:
```bash
# 对每个噪声通道，单独开启该通道运行 STIRAP 100 次 MC 评估
# 修改 NoiseModel 或 evaluate.py 支持单通道模式
python -c "
from src.baselines.evaluate import evaluate_policy
from src.baselines.stirap import run_stirap
# 分别对 5 个通道 + 全通道 运行评估
for channel in ['doppler', 'position', 'amplitude', 'phase', 'decay', 'all']:
    result = evaluate_single_channel(run_stirap, 'B', channel, n_traj=200)
    print(f'{channel}: 1-F = {1-result[\"mean_F\"]:.6f}')
"
```

### 2.2 Fig.08 — 传统算法天花板 vs 门时间 — **完全伪造**

**文件**: `src/plotting/fig08_traditional_ceiling.py`
**代码注释**: `"Uses analytic/synthetic curves."`
**问题**: STIRAP 和 GRAPE 的 F(T_gate) 曲线均为发明的解析公式：
```python
# 第 29 行：发明的 STIRAP 模型
F_stirap = (1 - (T_ad / T_s)**2) * np.exp(-T_s / tau_eff)
# T_ad = 0.15e-6 是随意选择的自由参数

# 第 35-36 行：发明的 GRAPE 模型
F_unitary = 1 - 0.0001 * np.exp(-T_us / 0.05)  # 0.0001 和 0.05 是随意系数
F_grape = F_unitary * np.exp(-T_s / tau_eff)
```

**如何真实验证**:
```bash
# 在 T_gate = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0] us 上
# 分别运行 STIRAP 和 GRAPE 优化+MC评估
python run_gate_time_sweep.py  # 需要新写
```

### 2.3 Fig.11 — 鲁棒性曲线 — **完全伪造**

**文件**: `src/plotting/fig11_robustness.py`
**代码注释**: `"Synthetic curves"`
**问题**: 三条 F(δΩ/Ω) 曲线全部是硬编码的线性/二次函数：
```python
F_stirap = 0.996 - 0.0006 * delta_pct          # 发明的线性模型
F_grape  = 0.999 - 0.96 * delta_frac**2         # 发明的二次模型
F_ppo    = 0.847 - 0.0003 * delta_pct           # 发明的线性模型
```
没有运行任何振幅扫描实验。

**如何真实验证**:
```bash
# 对 δΩ/Ω = 0%, 1%, 2%, 3%, 4%, 5%
# 分别用 STIRAP / GRAPE / PPO 策略评估
python run_robustness_sweep.py  # 需要新写
```

### 2.4 Fig.13 — 脉冲形状对比 — **PPO 部分伪造**

**文件**: `src/plotting/fig13_pulse_comparison.py`
**代码注释**: `"PPO uses a synthesized plausible learned pulse"`
**问题**: STIRAP 脉冲是合法的 sin² 解析公式，但 PPO 脉冲完全是编造的：
```python
# 第 44 行：发明的"类学习"脉冲形状
Omega_ppo_norm = np.sin(np.pi * tau) ** 1.4 * (1 + 0.15 * np.sin(2 * np.pi * tau))
# 第 51 行：发明的 chirp
Delta_ppo = 2 * np.pi * 0.3e6 * np.sin(2 * np.pi * tau) * np.exp(...)
```
指数 1.4、调制幅度 0.15、chirp 参数均为随意选择以"看起来像学到的"。

**如何真实验证**:
```bash
# 加载训练好的 PPO 模型，rollout 一个 episode，记录每步的 action
from stable_baselines3 import PPO
model = PPO.load("models/ppo_B_best")
env = RydbergBellEnv(scenario="B", n_steps=30, use_noise=False)
obs, _ = env.reset()
actions = []
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    actions.append(action)
    obs, _, done, _, _ = env.step(action)
# actions 即为真实 PPO 脉冲
```

### 2.5 Fig.14 — 布居数演化 — **PPO 部分伪造**

**文件**: `src/plotting/fig14_population_evolution.py`
**代码注释**: `"Right panel uses plausible synthesized PPO curves."`
**问题**:
- **STIRAP 面板**: 尝试通过 `run_stirap('B')` 真实模拟（第 27 行），但有合成数据后备（第 62-69 行）。如果 QuTiP 可用则为真实数据。
- **PPO 面板**: 完全编造（第 72-94 行），使用发明的指数/正弦函数，手动调整使末态 P_W ≈ 0.85。

**如何真实验证**:
```bash
# 加载 PPO 模型，在 env 中 rollout，每步记录 rho 并提取布居数
# 类似 fig13 的验证方法，但额外记录 density matrix
```

---

## 三、RL 环境与基线的简化/差异（未在报告中说明）

### 3.1 RL 环境噪声模型简化

**问题**: `src/environments/rydberg_env.py` 的噪声处理与 `src/physics/lindblad.py` 存在两处关键差异：

| 噪声通道 | `lindblad.py` (基线用) | `rydberg_env.py` (PPO 用) | 差异严重性 |
|---------|----------------------|--------------------------|----------|
| 振幅 OU 噪声 | 时间相关的 OU 过程 (τ_c = 10 μs) | **每步独立高斯** `N(0, σ)` | **高** — 丢失了时间相关性 |
| Doppler 频移 | 每个原子独立频移 Δ_i | **两原子平均** `(Δ₁+Δ₂)/2` | **中** — 丢失了差分效应 |
| 位置涨落 | 修改每对原子间距 R_eff | 不应用于 vdW 交互 | **中** |

**后果**: PPO 策略在训练时面对的噪声模型与基线评估时的噪声模型**不完全相同**。报告中说"三种方法在相同噪声模型下比较"是不严格准确的。

**修复方案**: 在 `rydberg_env.py` 中实现完整的 OU 过程和逐原子 Doppler。

### 3.2 "STIRAP" 命名不准确

**问题**: `src/baselines/stirap.py` 实现的是 **sin² 包络的共振 π 脉冲**（利用 blockade 的有效两能级系统），而非真正的 STIRAP（受激拉曼绝热通道，三能级 Λ 系统 + 反直觉脉冲序列）。

代码 docstring 写的是 `"STIRAP-like adiabatic pulse"`，但报告正文多处直接称之为 "STIRAP"，暗示了三能级绝热转移。

---

## 四、报告正文中的错误声明

### 4.1 数值计算错误

| 位置 | 报告声明 | 实际计算值 | 偏差 |
|------|---------|----------|------|
| §5.1 Eq.(5.1), 第 647 行 | $V_{\text{vdW}}/2\pi \approx 500$ MHz | **240.6 MHz** | **2.08× 夸大** |
| §5.1, 第 653 行 | blockade 比 $V_{\text{vdW}}/\Omega \approx 109$ | **52.3** | **2.08× 夸大** |
| §7 多处 | $\tau_{\text{eff}} = 88\;\mu\text{s}$ | `constants.py` 计算 **80.6 μs** | 9% 偏差 |

**验证**: `V = C6/R⁶ = 2π×15.4e9 / 2⁶ = 2π×240.6e6` rad/s。报告写 500 MHz 是错误的。

### 4.2 Monte Carlo 轨迹数虚报

| 位置 | 报告声明 | 实际执行 |
|------|---------|---------|
| §5.4, 第 725 行 | "执行 **1000 次** Monte Carlo 轨迹" | STIRAP/GRAPE: **100 次**, PPO: **200 次** |
| Table 3 标题 | "100 次 Monte Carlo" | 与结果文件一致 |

§5.4 的 "1000 次" 与 Table 3 的 "100 次" 自相矛盾。

### 4.3 场景 D — 从未执行，但报告多处引用

| 位置 | 声明 | 事实 |
|------|------|------|
| §5.3 Tab.2 | 场景 D: "传统方法 F < 0.95" | **从未计算** — 无 `*_D.json` 结果文件 |
| §5.3 正文 | "场景 D 将问题推向三原子 W 态" | 代码支持 3-atom Hamiltonian，但**从未运行** |
| `rydberg_env.py` 第 79 行 | — | `if n_atoms != 2: raise ValueError` — **RL 环境明确拒绝 3-atom** |

"传统方法 F < 0.95" 这一声明没有任何计算支持。

### 4.4 Cherry-picking PPO 结果

| 位置 | 报告说法 | 完整事实 |
|------|---------|---------|
| 摘要 | "PPO 达到 $F_{\text{train}} \approx 0.98$" | 仅 seed 264 达到 0.980；seed 42 = 0.925, seed 153 = 0.931 |
| §7.5 | "PPO 能从零学到量子控制（F 从 0.04 到 0.98）" | 0.98 是**训练**保真度 (stochastic policy)，**评估**保真度仅 0.847 |

### 4.5 对 GRAPE 失效的过度渲染

报告 §4.3(b) 花大量篇幅论述 GRAPE 的"开环脆弱性"和 barren plateau 问题，但：
- GRAPE 在场景 B 实际达到 **F = 0.996**，表现优异
- Appendix D 自己承认："对于小系统（如 2-qubit 门，d=4），barren plateau 不是问题"
- 报告描述 GRAPE 使用解析梯度（Appendix D 推导），但代码实际使用**数值有限差分梯度**
- "sim-to-real gap" 从未被测试——没有任何实验扰动评估模型与训练模型的差异

### 4.6 PPO 低方差的误导性解读

报告 §7.2(iv) 将 PPO 的 $\sigma_F = 0.004$ 解读为 "domain randomization 赋予的内在鲁棒性优势"。但在 $F = 0.847$ 的低保真度下，低方差可能仅仅说明策略**一致地表现平庸**，而非真正的鲁棒性。

### 4.7 未实施的分析声明

| 声明 | 位置 | 实际状况 |
|------|------|---------|
| "DR 策略自动调整 Δ(t) 补偿 Doppler 频移" | §6.3.3 | **从未分析学到的策略行为** |
| "DR 学习了自适应相位调制 (chirp) 和幅度调节" | §8.2 | **无策略检查、无频率分析** |
| "STA/CD 在非马尔可夫噪声下失效" | §4.3(c) | **无 counter-adiabatic driving 代码** |
| "expm 比 mesolve 快 5 倍" | §7.1 | **无基准测试数据** |
| "STIRAP F ≤ 0.985 Goldilocks 区间" | §4.3(a) | 场景 B 实际 F = 0.996，该声明未经验证 |

---

## 五、完整验证清单（按优先级排序）

### P0 — 必须修复的错误

- [ ] **修正 V_vdW 数值**: 报告中 500 MHz → 240.6 MHz, blockade 比 109 → 52.3
- [ ] **修正 τ_eff**: 报告中 88 μs → 80.6 μs（与代码一致）
- [ ] **修正 MC 轨迹数**: §5.4 的 "1000 次" → "100-200 次"（与实际一致）
- [ ] **删除或标注场景 D 性能声明**: "F < 0.95" 无计算支持

### P1 — 需要补充真实计算的图表

- [ ] **Fig.07**: 编写逐通道噪声扫描脚本，真实运行
- [ ] **Fig.08**: 编写门时间扫描脚本 (T_gate 从 0.1 到 10 μs)
- [ ] **Fig.11**: 编写振幅鲁棒性扫描脚本 (δΩ/Ω 从 0% 到 5%)
- [ ] **Fig.13 PPO 部分**: 从 `models/ppo_B_best.zip` 提取真实脉冲
- [ ] **Fig.14 PPO 部分**: 从模型 rollout 提取真实布居数演化

### P2 — 需要补充计算或重新表述的声明

- [ ] 场景 A 的 GRAPE 评估（当前仅有 STIRAP）
- [ ] 场景 A/D 的 PPO 训练与评估
- [ ] RL 环境噪声模型对齐（OU 时间相关性、逐原子 Doppler）
- [ ] "expm 5× 加速" 基准测试
- [ ] 学到的 PPO 策略的定性分析（脉冲形状、频率补偿行为）

### P3 — 需要在报告中明确说明的简化

- [ ] "STIRAP" 实际是 blockade-mediated π-pulse with sin² envelope
- [ ] RL 环境的 OU 噪声简化为 i.i.d.、Doppler 取平均
- [ ] GRAPE 使用数值梯度（非 Appendix D 的解析梯度）
- [ ] barren plateau 分析对 d=4 系统不适用
- [ ] PPO 50k 步为 demo 级训练预算

---

## 六、各验证任务的脚本模板

### 6.1 逐通道噪声扫描 (Fig.07)

```python
"""run_noise_channel_sweep.py — 逐个开启噪声通道评估 STIRAP"""
import json, numpy as np
from src.baselines.evaluate import evaluate_policy
from src.baselines.stirap import run_stirap
from src.physics.noise_model import NoiseModel

SCENARIO = "B"
N_TRAJ = 200
CHANNELS = ["doppler", "position", "amplitude", "phase", "decay"]

results = {}
for ch in CHANNELS + ["all"]:
    def run_fn(scenario, noise_params=None, **kw):
        # 只保留指定通道的噪声
        if ch != "all":
            filtered = {}
            nm = NoiseModel(scenario)
            full_noise = nm.sample(np.random.default_rng())
            # 仅保留 ch 对应的键
            # ... 需要实现通道过滤逻辑
            pass
        return run_stirap(scenario, noise_params=noise_params)
    
    res = evaluate_policy(run_fn, SCENARIO, n_trajectories=N_TRAJ)
    results[ch] = {"mean_F": res["mean_F"], "infidelity": 1 - res["mean_F"]}
    print(f"{ch}: 1-F = {1-res['mean_F']:.6f}")

with open("results/noise_channel_sweep_B.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 6.2 门时间扫描 (Fig.08)

```python
"""run_gate_time_sweep.py — 扫描 T_gate 评估 STIRAP 和 GRAPE"""
import json, numpy as np, copy
from src.physics.constants import SCENARIOS
from src.baselines.stirap import run_stirap
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy

T_gates = [0.1e-6, 0.2e-6, 0.3e-6, 0.5e-6, 1e-6, 2e-6, 5e-6, 10e-6]
N_TRAJ = 100
results = []

for T in T_gates:
    # 需要临时修改 SCENARIOS 或 传入 T_gate 参数
    # ... 实现参数覆盖逻辑
    pass
```

### 6.3 振幅鲁棒性扫描 (Fig.11)

```python
"""run_robustness_sweep.py — 扫描振幅偏差评估三种方法"""
# 对 delta_Omega/Omega = 0%, 1%, 2%, 3%, 4%, 5%
# 在噪声采样中添加系统性振幅偏移
# 分别评估 STIRAP, GRAPE, PPO
```

### 6.4 提取真实 PPO 脉冲 (Fig.13)

```python
"""extract_ppo_pulse.py — 从训练好的模型提取控制脉冲"""
from stable_baselines3 import PPO
from src.environments.rydberg_env import RydbergBellEnv
import numpy as np, json

model = PPO.load("models/ppo_B_best")
env = RydbergBellEnv(scenario="B", n_steps=30, use_noise=False)
obs, _ = env.reset(seed=0)

omegas, deltas = [], []
for step in range(30):
    action, _ = model.predict(obs, deterministic=True)
    # 解码 action → 物理参数
    a = np.clip(action, -1, 1)
    Omega = (a[0] + 1) / 2 * 2 * env.Omega_max
    Delta = a[1] * env.Omega_max
    omegas.append(float(Omega))
    deltas.append(float(Delta))
    obs, _, done, _, info = env.step(action)

with open("results/ppo_pulse_B.json", "w") as f:
    json.dump({"omega": omegas, "delta": deltas}, f, indent=2)
print(f"Final fidelity: {info.get('fidelity', 'N/A')}")
```

### 6.5 提取真实 PPO 布居数演化 (Fig.14)

```python
"""extract_ppo_populations.py — 从模型 rollout 提取逐步布居数"""
# 修改 rydberg_env.py 的 step() 在 info 中返回当前 rho
# 或在 rollout 后重放脉冲序列，逐步记录 <gg|rho|gg>, <W|rho|W>, <rr|rho|rr>
```

---

## 七、总结

本项目的**物理模拟内核是正确的**，**PPO 训练和基线评估确实运行了**，核心数值结果（Table 3, 4）有真实计算支持。

但项目存在以下诚信问题：
1. **5 幅图表使用合成数据**，报告正文未标注
2. **报告中存在明确的数值计算错误**（V_vdW 夸大 2 倍）
3. **场景 D 从未执行任何计算**，但被当作实验条件引用
4. **对 GRAPE 失效的论述与实际结果矛盾**
5. **PPO 结果存在 cherry-picking**（摘要引用最佳 seed）
6. **RL 环境噪声模型与基线使用的噪声模型不完全一致**

这些问题需要在提交报告前修复。优先级 P0（数值错误）应立即修正；P1（图表）应通过真实计算替换合成数据；P2-P3（补充实验/说明简化）可视时间预算处理。

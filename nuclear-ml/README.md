# 核结构机器学习：基于物理先验的结合能预测

**作者：** 应嘉禾 (230012440, 物理学院)  
**项目仓库：** https://github.com/starpacker/rydberg-rl-entanglement

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 项目概述

本项目探索机器学习在核结合能预测中的应用，重点研究**物理先验**在小样本学习中的关键作用。基于AME2020实验数据库，我们系统比较了三种方法：

1. **直接回归**：纯数据驱动，完全失败（RMSE 33.25 MeV）
2. **残差学习（SEMF+NN）**：物理先验+神经网络修正，**最佳性能**（RMSE 3.14 MeV，相对SEMF改进67.2%）
3. **哈密顿量预测**：SO(3)等变网络，因架构问题未能收敛

## 核心成果

| 方法 | RMSE (MeV) | MAE (MeV) | 相对SEMF改进 | 参数量 |
|------|------------|-----------|--------------|--------|
| SEMF基线 | 9.57 | 5.70 | — | 4 |
| 直接回归 | 33.25 | 18.94 | **-247%** ❌ | 121K |
| **残差学习** | **3.14** | **1.77** | **+67.2%** ✅ | 27K |
| 哈密顿量 | — | — | 失败 ❌ | 41K |

**数据集：** AME2020，243个轻核（Z=2-20, N=2-28），训练/验证/测试 = 162/28/53

---

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n nuclear-ml python=3.10
conda activate nuclear-ml

# 安装依赖
pip install torch numpy scipy matplotlib pandas pyyaml
pip install e3nn  # 仅用于等变网络实验
```

### 2. 数据准备

```bash
# 下载AME2020数据
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt -P data/raw/

# 预处理数据
python scripts/prepare_ame2020_data.py
```

### 3. 训练模型

```bash
# 训练残差学习模型（推荐）
python training/train.py --config configs/ame2020_residual.yaml

# 训练直接回归模型（对比）
python training/train.py --config configs/ame2020_direct.yaml
```

### 4. 评估结果

```bash
# 计算SEMF基线
python scripts/compute_semf_baseline.py

# 评估所有模型
python scripts/evaluate_ame2020_models.py

# 可视化结果
python scripts/visualize_results.py
```

---

## 项目结构

```
nuclear-ml/
├── README.md                    # 本文件
├── PROJECT_SUMMARY.md           # 项目总结与经验教训
│
├── configs/                     # 训练配置文件
│   ├── ame2020_direct.yaml      # 直接回归配置
│   ├── ame2020_residual.yaml    # 残差学习配置（最佳）
│   ├── ame2020_equivariant.yaml # 等变网络配置
│   └── ame2020_simple.yaml      # 简单哈密顿量网络
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始AME2020数据
│   │   └── mass_1.mas20.txt     # AME2020质量表
│   └── processed/               # 处理后的数据
│       ├── ame2020_phase_a_train.json
│       ├── ame2020_phase_a_val.json
│       └── ame2020_phase_a_test.json
│
├── models/                      # 模型定义
│   ├── direct_regression.py     # 直接回归 & 残差学习模型
│   ├── hamiltonian_simple.py    # 简单哈密顿量网络
│   ├── hamiltonian_equivariant.py # SO(3)等变网络
│   └── solver.py                # 可微分本征值求解器
│
├── training/                    # 训练脚本
│   └── train.py                 # 统一训练入口
│
├── scripts/                     # 实用脚本
│   ├── prepare_ame2020_data.py  # 数据预处理
│   ├── evaluate_ame2020_models.py # 模型评估
│   ├── compute_semf_baseline.py # SEMF基线计算
│   └── visualize_results.py     # 结果可视化
│
├── results/                     # 实验结果
│   ├── ame2020_residual/        # 残差学习结果（最佳）
│   │   ├── best_model.pt
│   │   ├── metrics.json
│   │   └── training_curves.png
│   ├── ame2020_direct/          # 直接回归结果
│   ├── ame2020_comparison/      # 方法对比
│   │   └── metrics.json
│   └── baselines/               # SEMF基线
│       └── semf_evaluation.json
│
├── latex/                       # LaTeX报告
│   ├── main_ame2020.tex         # 主文件
│   ├── main_ame2020.pdf         # 最终PDF（19页）
│   └── refs_ame2020.bib         # 参考文献
│
├── logs/                        # 训练日志
│   ├── ame2020_residual.log
│   ├── ame2020_direct.log
│   └── ame2020_equivariant.log
│
├── docs/                        # 文档
│   ├── AME2020_TRAINING_STATUS.md
│   └── STATUS_AND_RECOMMENDATIONS.md
│
└── notebooks/                   # Jupyter notebooks（探索性分析）
```

---

## 核心发现

### 1. 物理先验至关重要

在小样本场景（243个核素）下：
- ✅ **残差学习**：SEMF提供物理先验，神经网络学习修正 → 成功
- ❌ **直接回归**：抛弃物理知识，纯数据驱动 → 完全失败
- ❌ **哈密顿量**：物理约束过强，架构设计问题 → 未能收敛

**结论：** 中等强度的物理先验（残差学习）是最优选择。

### 2. 参数效率

- 残差模型：27K参数，RMSE 3.14 MeV ✅
- 直接回归：121K参数，RMSE 33.25 MeV ❌

**更多参数 ≠ 更好性能**。物理先验降低了学习难度，需要更少参数。

### 3. 问题简化

残差学习将问题从"学习B(Z,N) ∈ [0, 400] MeV"简化为"学习ΔB ∈ [-30, +30] MeV"：
- 动态范围缩小6.7倍
- 函数更平滑（梯度范数降低7.3倍）
- 更容易泛化

### 4. 架构设计教训

**哈密顿量方法失败的原因：**
1. **维度灾难**：N×N矩阵有N²自由度，数据不足
2. **梯度消失**：本征值求解器的梯度在深层网络中消失
3. **架构不匹配**：SO(3)等变输出与Hermitian矩阵构造不兼容

**启示：** 物理优雅性 ≠ 实用性。必须平衡理论严格性与计算可行性。

---

## 实验复现

### 完整实验流程

```bash
# 1. 数据准备
python scripts/prepare_ame2020_data.py

# 2. 计算SEMF基线
python scripts/compute_semf_baseline.py

# 3. 训练残差模型
python training/train.py --config configs/ame2020_residual.yaml

# 4. 训练直接回归（对比）
python training/train.py --config configs/ame2020_direct.yaml

# 5. 评估所有模型
python scripts/evaluate_ame2020_models.py

# 6. 生成可视化
python scripts/visualize_results.py
```

### 预期结果

训练完成后，`results/ame2020_comparison/metrics.json`应包含：

```json
{
  "test_samples": 53,
  "semf": {
    "rmse": 9.573,
    "mae": 5.695
  },
  "models": {
    "Residual (SEMF+NN)": {
      "rmse": 3.140,
      "mae": 1.775
    },
    "Direct Regression": {
      "rmse": 33.248,
      "mae": 18.938
    }
  }
}
```

---

## 报告生成

### LaTeX编译

```bash
cd latex/

# 使用tectonic（推荐）
tectonic main_ame2020.tex

# 或使用XeLaTeX
xelatex main_ame2020.tex
bibtex main_ame2020
xelatex main_ame2020.tex
xelatex main_ame2020.tex
```

生成的PDF包含：
- 15页正文（引言、方法、结果、讨论、结论）
- 4页附录（数据统计、超参数、SEMF推导）
- 完整参考文献（6篇核心文献）

---

## 经验教训

### ✅ 成功经验

1. **物理先验是小样本学习的关键**
   - 在有限数据下，物理知识不是可选项，而是必需品
   - SEMF提供正确的标度和趋势，神经网络只需学习修正

2. **残差学习降低学习难度**
   - 原问题：学习B(Z,N) ∈ [0, 400] MeV
   - 残差问题：学习ΔB ∈ [-30, +30] MeV
   - 动态范围缩小 → 更容易泛化

3. **参数效率优于参数数量**
   - 27K参数的残差模型 > 121K参数的直接回归
   - 物理先验 = 强归纳偏置 = 更少参数

4. **系统对比揭示本质**
   - 对比无先验、中等先验、强先验三种方法
   - 清晰展示物理先验的作用

### ❌ 失败教训

1. **直接回归在小样本下完全失效**
   - 121K参数 vs 162训练样本 = 参数/数据比 748:1
   - 模型记忆噪声而非学习规律
   - **教训：** 数据不足时，必须引入先验知识

2. **哈密顿量方法的三重困境**
   - 维度灾难：N²自由度 > 数据量
   - 梯度消失：本征值求解器梯度不稳定
   - 架构不匹配：等变输出 → Hermitian矩阵转换困难
   - **教训：** 物理优雅性必须与计算可行性平衡

3. **过度工程化的风险**
   - SO(3)等变网络理论完美，但实现复杂
   - 简单的残差学习反而效果最好
   - **教训：** 奥卡姆剃刀原则——简单方法优先

4. **数据质量 > 数据量**
   - 使用SEMF生成的"伪数据"训练效果有限
   - 真实实验数据（AME2020）才是关键
   - **教训：** 高质量小数据 > 低质量大数据

---

## 参考文献

1. Wang et al. (2021). The AME2020 atomic mass evaluation. *Chinese Physics C*, 45(3), 030003.
2. Weizsäcker (1935). Zur Theorie der Kernmassen. *Zeitschrift für Physik*, 96(7-8), 431-458.
3. Möller et al. (2016). Nuclear ground-state masses and deformations: FRDM(2012). *Atomic Data and Nuclear Data Tables*, 109, 1-204.
4. Niu et al. (2018). Nuclear mass predictions based on Bayesian neural network approach. *Physics Letters B*, 778, 48-53.
5. Utama et al. (2016). Nuclear mass predictions for neutron stars: A Bayesian neural network approach. *Physical Review C*, 93(1), 014311.

---

## 许可证

MIT License

---

## 联系方式

- **作者：** 应嘉禾
- **学号：** 230012440
- **单位：** 物理学院
- **项目仓库：** https://github.com/starpacker/rydberg-rl-entanglement

---

**最后更新：** 2026-05-26

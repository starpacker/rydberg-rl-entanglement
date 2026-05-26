# 目录结构详细说明

本文档详细说明项目中每个目录和重要文件的用途。

---

## 顶层目录

```
nuclear-ml/
├── README.md                    # 项目概述和快速开始指南
├── PROJECT_SUMMARY.md           # 项目总结、经验教训、完整实验记录
├── DIRECTORY_GUIDE.md           # 本文件：目录结构详细说明
├── configs/                     # 训练配置文件（YAML格式）
├── data/                        # 数据目录
├── models/                      # 模型定义
├── training/                    # 训练脚本
├── scripts/                     # 实用脚本
├── results/                     # 实验结果
├── latex/                       # LaTeX报告
├── logs/                        # 训练日志
├── docs/                        # 项目文档
└── notebooks/                   # Jupyter notebooks
```

---

## configs/ - 训练配置文件

所有配置文件使用YAML格式，包含模型、训练、数据等参数。

### 文件列表

- **`ame2020_residual.yaml`** ✅ **推荐使用**
  - 残差学习模型配置
  - 最佳性能：RMSE 3.14 MeV
  - 参数：128-128-64隐藏层，dropout 0.1

- **`ame2020_direct.yaml`**
  - 直接回归模型配置（对比基线）
  - 性能差：RMSE 33.25 MeV
  - 参数：256-256-128-128隐藏层，dropout 0.2

- **`ame2020_equivariant.yaml`**
  - SO(3)等变网络配置
  - 未能成功训练

- **`ame2020_simple.yaml`**
  - 简单哈密顿量网络配置
  - 早期实验使用

### 配置文件格式示例

```yaml
model:
  type: residual              # 模型类型
  hidden_dims: [128, 128, 64] # 隐藏层维度
  dropout: 0.1                # Dropout率
  activation: silu            # 激活函数

training:
  epochs: 500                 # 训练轮数
  batch_size: 32              # 批大小
  learning_rate: 0.001        # 学习率
  weight_decay: 0.00001       # L2正则化
  early_stopping_patience: 20 # 早停耐心值

data:
  train_path: data/processed/ame2020_phase_a_train.json
  val_path: data/processed/ame2020_phase_a_val.json
  test_path: data/processed/ame2020_phase_a_test.json
```

---

## data/ - 数据目录

### 子目录结构

```
data/
├── raw/                        # 原始数据（不修改）
│   └── mass_1.mas20.txt        # AME2020原始质量表
└── processed/                  # 处理后的数据（JSON格式）
    ├── ame2020_phase_a_train.json  # 训练集（162样本）
    ├── ame2020_phase_a_val.json    # 验证集（28样本）
    └── ame2020_phase_a_test.json   # 测试集（53样本）
```

### 数据格式

**原始数据（mass_1.mas20.txt）：**
- 来源：https://www-nds.iaea.org/amdc/
- 格式：固定宽度文本文件
- 内容：核素质量、结合能、不确定度等

**处理后数据（JSON）：**
```json
[
  {
    "Z": 2,
    "N": 2,
    "A": 4,
    "binding_energy": 28.295673,
    "element": "He"
  },
  ...
]
```

### 数据统计

| 数据集 | 样本数 | Z范围 | N范围 | A范围 | 结合能范围 (MeV) |
|--------|--------|-------|-------|-------|------------------|
| 训练集 | 162 | 2-20 | 2-28 | 4-48 | 2.2-411.5 |
| 验证集 | 28 | 2-20 | 2-28 | 4-48 | 7.7-398.3 |
| 测试集 | 53 | 2-20 | 2-28 | 4-48 | 14.4-407.9 |
| **总计** | **243** | **2-20** | **2-28** | **4-48** | **2.2-411.5** |

---

## models/ - 模型定义

### 文件列表

1. **`direct_regression.py`** ✅ **核心文件**
   - `DirectRegressionNet`：直接回归模型（失败案例）
   - `ResidualRegressionNet`：残差学习模型（最佳）
   - `compute_semf_features()`：计算物理特征
   - `compute_semf_binding_energy()`：计算SEMF基线

2. **`hamiltonian_simple.py`**
   - `SimpleHamiltonianNet`：简单哈密顿量网络
   - 直接预测哈密顿量矩阵元
   - 早期实验使用，未采用

3. **`hamiltonian_equivariant.py`**
   - `EquivariantHamiltonianNet`：SO(3)等变网络
   - 使用e3nn库实现
   - 理论优雅但实践失败

4. **`solver.py`**
   - `DifferentiableSolver`：可微分本征值求解器
   - 支持梯度回传
   - 用于哈密顿量方法

### 模型架构对比

| 模型 | 输入 | 输出 | 参数量 | 特点 |
|------|------|------|--------|------|
| DirectRegressionNet | 17维特征 | 结合能 | 121K | 纯数据驱动 |
| ResidualRegressionNet | 17维特征 | 残差修正 | 27K | SEMF+NN |
| SimpleHamiltonianNet | (Z,N) | N×N矩阵 | 500K | 物理约束 |
| EquivariantHamiltonianNet | (Z,N) | 球谐系数 | 41K | SO(3)对称 |

---

## training/ - 训练脚本

### 文件列表

- **`train.py`** ✅ **主训练脚本**
  - 统一训练入口
  - 支持所有模型类型
  - 自动保存最佳模型
  - 生成训练曲线

### 使用方法

```bash
# 训练残差学习模型（推荐）
python training/train.py --config configs/ame2020_residual.yaml

# 训练直接回归模型（对比）
python training/train.py --config configs/ame2020_direct.yaml

# 指定输出目录
python training/train.py \
  --config configs/ame2020_residual.yaml \
  --output results/my_experiment
```

### 训练输出

训练脚本会在输出目录生成：
- `best_model.pt`：最佳模型权重
- `final_model.pt`：最终模型权重
- `metrics.json`：训练和验证指标
- `training_curves.png`：损失曲线图
- `config.yaml`：使用的配置文件副本

---

## scripts/ - 实用脚本

### 文件列表

1. **`prepare_ame2020_data.py`** ✅ **数据预处理**
   - 从AME2020原始文件生成训练/验证/测试集
   - 按Z分层抽样
   - 输出JSON格式

   ```bash
   python scripts/prepare_ame2020_data.py
   ```

2. **`compute_semf_baseline.py`** ✅ **基线计算**
   - 计算SEMF在测试集上的性能
   - 生成基线评估报告

   ```bash
   python scripts/compute_semf_baseline.py
   ```

3. **`evaluate_ame2020_models.py`** ✅ **模型评估**
   - 评估所有训练好的模型
   - 生成对比报告
   - 输出到`results/ame2020_comparison/`

   ```bash
   python scripts/evaluate_ame2020_models.py
   ```

4. **`visualize_results.py`** ✅ **结果可视化**
   - 生成预测散点图
   - 残差分布直方图
   - 误差分析图表

   ```bash
   python scripts/visualize_results.py
   ```

---

## results/ - 实验结果

### 目录结构

```
results/
├── ame2020_residual/           # 残差学习结果（最佳）
│   ├── best_model.pt           # 最佳模型权重
│   ├── metrics.json            # 性能指标
│   └── training_curves.png     # 训练曲线
│
├── ame2020_direct/             # 直接回归结果
│   ├── best_model.pt
│   ├── metrics.json
│   └── training_curves.png
│
├── ame2020_comparison/         # 方法对比
│   ├── metrics.json            # 所有方法的对比指标
│   ├── comparison_plot.png     # 对比图表
│   └── residual_analysis.png   # 残差分析
│
├── baselines/                  # 基线方法
│   └── semf_evaluation.json    # SEMF基线性能
│
└── [其他实验目录]/             # 早期实验结果
    ├── phase_a/
    ├── phase_a_equivariant/
    └── multiseed/
```

### 结果文件格式

**metrics.json 示例：**
```json
{
  "train": {
    "loss": 2.456,
    "rmse": 1.567,
    "mae": 0.892
  },
  "val": {
    "loss": 3.123,
    "rmse": 1.767,
    "mae": 1.012
  },
  "test": {
    "loss": 9.876,
    "rmse": 3.142,
    "mae": 1.775,
    "max_error": 16.79
  },
  "model_params": 27905,
  "training_time": 49.5
}
```

---

## latex/ - LaTeX报告

### 文件列表

- **`main_ame2020.tex`** ✅ **主文件**
  - 19页完整报告
  - 包含引言、方法、结果、讨论、结论
  - 附录：数据统计、超参数、SEMF推导

- **`main_ame2020.pdf`** ✅ **最终PDF**
  - 编译后的PDF文件
  - 可直接提交

- **`refs_ame2020.bib`** ✅ **参考文献**
  - 6篇核心文献
  - BibTeX格式

### 编译方法

```bash
cd latex/

# 方法1：使用tectonic（推荐）
tectonic main_ame2020.tex

# 方法2：使用XeLaTeX
xelatex main_ame2020.tex
bibtex main_ame2020
xelatex main_ame2020.tex
xelatex main_ame2020.tex
```

### 报告结构

1. **摘要**（1页）
2. **引言**（2页）
   - 研究背景
   - 研究动机
   - 研究目标
3. **理论与方法**（4页）
   - SEMF公式
   - 直接回归
   - 残差学习
   - 哈密顿量方法
4. **数据与实验设置**（2页）
   - AME2020数据集
   - 数据划分
   - 训练配置
5. **实验结果**（3页）
   - 定量结果
   - 方法对比
   - 误差分析
6. **讨论**（2页）
   - 物理先验的作用
   - 失败案例分析
   - 与文献对比
7. **结论**（1页）
8. **参考文献**（1页）
9. **附录**（4页）
   - A.1 数据集统计
   - A.2 超参数搜索
   - A.3 计算资源
   - A.4 代码可用性
   - A.5 SEMF推导

---

## logs/ - 训练日志

### 文件列表

- `ame2020_residual.log`：残差学习训练日志
- `ame2020_direct.log`：直接回归训练日志
- `ame2020_equivariant.log`：等变网络训练日志

### 日志格式

```
2026-05-25 14:23:45 - INFO - Starting training...
2026-05-25 14:23:45 - INFO - Model: ResidualRegressionNet
2026-05-25 14:23:45 - INFO - Parameters: 27905
2026-05-25 14:23:46 - INFO - Epoch 1/500 - Train Loss: 45.23 - Val Loss: 52.34
2026-05-25 14:23:47 - INFO - Epoch 2/500 - Train Loss: 38.12 - Val Loss: 45.67
...
2026-05-25 14:24:35 - INFO - Best model saved at epoch 234
2026-05-25 14:24:35 - INFO - Training completed in 49.5s
```

---

## docs/ - 项目文档

### 文件列表

- **`AME2020_TRAINING_STATUS.md`**
  - AME2020实验的训练状态记录
  - 包含各模型的训练进度和结果

- **`STATUS_AND_RECOMMENDATIONS.md`**
  - 项目状态总结
  - 下一步建议

### 历史文档（早期实验）

- `DEVLOG.md`：开发日志
- `TODO.md`：任务列表
- `week1_day1_summary.md`：第一天总结

---

## notebooks/ - Jupyter Notebooks

用于探索性数据分析和结果可视化。

### 典型内容

- 数据分布分析
- 特征相关性分析
- 模型预测可视化
- 误差分析
- 超参数调优实验

---

## 其他重要文件

### 根目录文件

- **`.gitignore`**
  - Git忽略规则
  - 排除大文件、临时文件、模型权重等

- **`requirements.txt`** (如果存在)
  - Python依赖包列表
  - 用于`pip install -r requirements.txt`

- **`environment.yml`** (如果存在)
  - Conda环境配置
  - 用于`conda env create -f environment.yml`

---

## 文件命名规范

### 配置文件
- 格式：`{dataset}_{model_type}.yaml`
- 示例：`ame2020_residual.yaml`

### 数据文件
- 格式：`{dataset}_{phase}_{split}.json`
- 示例：`ame2020_phase_a_train.json`

### 结果目录
- 格式：`{dataset}_{model_type}/`
- 示例：`ame2020_residual/`

### 日志文件
- 格式：`{dataset}_{model_type}.log`
- 示例：`ame2020_residual.log`

---

## 快速导航

### 我想...

**训练一个新模型**
1. 准备配置文件：`configs/my_config.yaml`
2. 运行：`python training/train.py --config configs/my_config.yaml`
3. 查看结果：`results/my_experiment/`

**评估现有模型**
1. 运行：`python scripts/evaluate_ame2020_models.py`
2. 查看：`results/ame2020_comparison/metrics.json`

**生成报告**
1. 编辑：`latex/main_ame2020.tex`
2. 编译：`cd latex && tectonic main_ame2020.tex`
3. 查看：`latex/main_ame2020.pdf`

**理解实验结果**
1. 阅读：`PROJECT_SUMMARY.md`
2. 查看：`results/ame2020_comparison/`
3. 参考：`latex/main_ame2020.pdf`

**复现实验**
1. 准备数据：`python scripts/prepare_ame2020_data.py`
2. 训练模型：`python training/train.py --config configs/ame2020_residual.yaml`
3. 评估结果：`python scripts/evaluate_ame2020_models.py`

---

## 磁盘使用

### 各目录大小估计

| 目录 | 大小 | 说明 |
|------|------|------|
| `data/raw/` | ~1 MB | AME2020原始数据 |
| `data/processed/` | ~100 KB | JSON格式数据 |
| `models/` | ~50 KB | Python代码 |
| `results/` | ~50 MB | 所有实验结果 |
| `latex/` | ~2 MB | LaTeX源文件和PDF |
| `logs/` | ~1 MB | 训练日志 |
| **总计** | **~55 MB** | 不含模型权重 |

### 清理建议

**可以删除的文件：**
- `results/phase_a*/`：早期实验结果
- `results/multiseed*/`：多随机种子实验
- `logs/*.log`：旧的训练日志

**必须保留的文件：**
- `results/ame2020_residual/`：最佳模型
- `results/ame2020_comparison/`：对比结果
- `latex/main_ame2020.pdf`：最终报告
- `data/processed/`：处理后的数据

---

**最后更新：** 2026-05-26

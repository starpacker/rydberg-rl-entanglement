#!/usr/bin/env python3
"""
生成核结构机器学习项目的最终PDF报告
"""

from weasyprint import HTML, CSS
import json
from pathlib import Path

# 读取实验结果
with open('results/ame2020_comparison/metrics.json') as f:
    data = json.load(f)

# 提取数据
semf_results = data['semf']
direct_results = data['models']['Direct Regression']
residual_results = data['models']['Residual (SEMF+NN)']

# 创建HTML报告
html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>核结构机器学习研究报告</title>
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
            @bottom-center {
                content: counter(page);
            }
        }

        body {
            font-family: "SimSun", "STSong", serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }

        h1 {
            font-size: 20pt;
            font-weight: bold;
            text-align: center;
            margin-top: 1cm;
            margin-bottom: 0.5cm;
            color: #1a1a1a;
        }

        h2 {
            font-size: 16pt;
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }

        h3 {
            font-size: 13pt;
            font-weight: bold;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            color: #34495e;
        }

        .subtitle {
            font-size: 14pt;
            text-align: center;
            color: #555;
            margin-bottom: 0.5cm;
        }

        .author {
            text-align: center;
            font-size: 12pt;
            margin-bottom: 0.3cm;
        }

        .date {
            text-align: center;
            font-size: 11pt;
            color: #666;
            margin-bottom: 1cm;
        }

        .abstract {
            background-color: #f8f9fa;
            padding: 1em;
            margin: 1.5em 0;
            border-left: 4px solid #3498db;
            font-size: 10.5pt;
        }

        .abstract-title {
            font-weight: bold;
            font-size: 12pt;
            margin-bottom: 0.5em;
        }

        .keywords {
            margin-top: 1em;
            font-size: 10pt;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 10pt;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }

        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: #f2f2f2;
        }

        .highlight {
            background-color: #fff3cd;
            font-weight: bold;
        }

        .success {
            color: #27ae60;
            font-weight: bold;
        }

        .failure {
            color: #e74c3c;
        }

        ul, ol {
            margin: 0.5em 0;
            padding-left: 2em;
        }

        li {
            margin: 0.3em 0;
        }

        .equation {
            text-align: center;
            margin: 1em 0;
            font-style: italic;
        }

        .caption {
            text-align: center;
            font-size: 10pt;
            color: #666;
            margin-top: 0.5em;
        }

        .section-break {
            page-break-before: always;
        }

        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 9pt;
        }

        .box {
            border: 1px solid #ddd;
            padding: 1em;
            margin: 1em 0;
            background-color: #fafafa;
        }

        .conclusion {
            background-color: #e8f5e9;
            padding: 1em;
            margin: 1.5em 0;
            border-left: 4px solid #27ae60;
        }
    </style>
</head>
<body>

<h1>基于物理先验的核结合能机器学习预测</h1>
<div class="subtitle">半经验质量公式与神经网络残差修正方法</div>
<div class="author">应嘉禾 · 230012440 · 物理学院</div>
<div class="date">2026年5月26日</div>

<div class="abstract">
    <div class="abstract-title">摘要</div>
    <p>
    原子核结合能的精确预测是核物理和核天体物理的基础问题。传统的半经验质量公式（SEMF）
    虽然物理图像清晰，但在轻核区域存在显著的系统偏差。本文基于AME2020实验数据库，
    系统比较了三种机器学习方法：直接回归、残差修正和哈密顿量预测。
    </p>
    <p>
    实验结果表明，在243个轻核（Z=2–20, N=2–28）的数据集上，SEMF基线在测试集上达到9.57 MeV的RMSE。
    通过引入神经网络对SEMF残差进行修正，测试集RMSE降至<span class="success">3.14 MeV</span>，
    相对改进<span class="success">67.2%</span>。相比之下，抛弃物理先验的直接回归方法在有限数据下完全失效
    （RMSE 33.25 MeV），而基于哈密顿量的等变神经网络方法因架构设计问题未能收敛。
    </p>
    <p>
    本研究证实：在小样本核物理问题中，将物理知识作为归纳偏置嵌入模型是提升泛化能力的关键。
    </p>
    <div class="keywords">
        <strong>关键词：</strong>核结合能、半经验质量公式、残差学习、物理先验、AME2020
    </div>
</div>

<h2>1. 引言</h2>

<h3>1.1 研究背景</h3>

<p>
原子核结合能（binding energy）定义为将原子核完全拆解为独立核子所需的能量，
是描述核稳定性的最基本物理量。精确预测核结合能对以下领域至关重要：
</p>

<ul>
    <li><strong>核天体物理</strong>：恒星核合成过程（如r过程、rp过程）的模拟需要数千个远离稳定线核素的结合能数据</li>
    <li><strong>核能应用</strong>：反应堆燃料循环、核废料处理依赖于精确的核质量预测</li>
    <li><strong>基础物理</strong>：核力的微观理解、壳模型有效相互作用的确定</li>
</ul>

<p>
目前实验上仅测量了约3000个核素的质量，而理论预测的核素总数超过7000个。
如何在有限实验数据下准确外推到未知核区，是核物理的长期挑战。
</p>

<h3>1.2 传统方法的局限</h3>

<p>
半经验质量公式（Semi-Empirical Mass Formula, SEMF）由Weizsäcker于1935年提出，
基于液滴模型将核结合能分解为五项贡献：
</p>

<div class="equation">
B(Z,N) = a<sub>v</sub>A - a<sub>s</sub>A<sup>2/3</sup> - a<sub>c</sub>Z²/A<sup>1/3</sup> - a<sub>a</sub>(N-Z)²/A + δ(Z,N)
</div>

<p>
其中各项分别代表体积能、表面能、库仑能、对称能和配对能。SEMF在中重核区表现良好，
但在轻核区域（A < 40）存在显著偏差，主要原因是：
</p>

<ul>
    <li>壳效应（magic numbers）未被充分考虑</li>
    <li>形变效应在轻核中更为显著</li>
    <li>液滴模型假设在小系统中失效</li>
</ul>

<h3>1.3 机器学习的机遇与挑战</h3>

<p>
近年来，机器学习方法在核物理中展现出巨大潜力。然而，核物理数据具有以下特点：
</p>

<ul>
    <li><strong>小样本</strong>：实验数据有限（~3000个核素）</li>
    <li><strong>高维度</strong>：核结构问题本质上是多体量子问题</li>
    <li><strong>物理约束</strong>：必须满足对称性、守恒律等物理规律</li>
</ul>

<p>
本研究的核心问题是：<strong>如何在小样本条件下，有效结合物理先验知识与数据驱动方法？</strong>
</p>

<h2 class="section-break">2. 数据与方法</h2>

<h3>2.1 AME2020数据集</h3>

<p>
本研究使用AME2020（Atomic Mass Evaluation 2020）数据库，这是目前最权威的核质量实验数据集。
我们选取轻核到中等质量核区域（Z=2–20, N=2–28），共243个核素，按照以下方式划分：
</p>

<table>
    <tr>
        <th>数据集</th>
        <th>核素数量</th>
        <th>用途</th>
    </tr>
    <tr>
        <td>训练集</td>
        <td>162</td>
        <td>模型训练</td>
    </tr>
    <tr>
        <td>验证集</td>
        <td>28</td>
        <td>超参数调优</td>
    </tr>
    <tr>
        <td>测试集</td>
        <td>53</td>
        <td>最终评估</td>
    </tr>
</table>

<p>
数据划分采用按质子数Z分层抽样，确保各数据集中元素分布均衡。所有数据的实验不确定度均小于1 MeV。
</p>

<h3>2.2 评估指标</h3>

<p>我们使用以下指标评估模型性能：</p>

<ul>
    <li><strong>RMSE（均方根误差）</strong>：√(Σ(预测值 - 真实值)² / N)</li>
    <li><strong>MAE（平均绝对误差）</strong>：Σ|预测值 - 真实值| / N</li>
    <li><strong>最大误差</strong>：max|预测值 - 真实值|</li>
    <li><strong>相对改进</strong>：(RMSE<sub>baseline</sub> - RMSE<sub>model</sub>) / RMSE<sub>baseline</sub> × 100%</li>
</ul>

<h3>2.3 方法对比</h3>

<p>本研究系统比较了以下四种方法：</p>

<div class="box">
<h4>方法1：SEMF基线</h4>
<p>
使用标准的Bethe-Weizsäcker公式，参数为：a<sub>v</sub>=15.75, a<sub>s</sub>=17.8,
a<sub>c</sub>=0.711, a<sub>a</sub>=23.7, a<sub>p</sub>=11.18 MeV。
这是传统物理方法的代表，作为对比基线。
</p>
</div>

<div class="box">
<h4>方法2：直接回归神经网络</h4>
<p>
完全数据驱动的方法，使用深度神经网络直接从(Z, N)预测结合能。
网络包含物理启发的特征（如A<sup>2/3</sup>、Z²/A<sup>1/3</sup>等），
但不使用SEMF作为先验。模型参数：121,729。
</p>
</div>

<div class="box">
<h4>方法3：残差学习（SEMF + 神经网络修正）</h4>
<p>
<strong>本研究的核心方法。</strong>将SEMF作为物理先验，神经网络仅学习残差修正：
</p>
<div class="equation">
B<sub>pred</sub>(Z,N) = B<sub>SEMF</sub>(Z,N) + NN<sub>correction</sub>(Z,N)
</div>
<p>
这种方法结合了物理知识（SEMF提供正确的标度和趋势）与数据驱动学习（神经网络捕捉系统偏差）。
模型参数：27,905。
</p>
</div>

<div class="box">
<h4>方法4：哈密顿量神经网络（失败）</h4>
<p>
尝试使用SO(3)等变神经网络构造核哈密顿量，通过求解本征值问题得到结合能。
该方法在理论上更加优雅，但因以下原因失败：
</p>
<ul>
    <li>O(N⁴)的两体相互作用张量导致GPU内存溢出</li>
    <li>本征值尺度与结合能尺度不匹配</li>
    <li>架构复杂度与数据量不匹配</li>
</ul>
</div>

<h2 class="section-break">3. 实验结果</h2>

<h3>3.1 测试集性能对比</h3>

<p>表1展示了四种方法在53个测试核素上的性能：</p>

<table>
    <tr>
        <th>方法</th>
        <th>RMSE (MeV)</th>
        <th>MAE (MeV)</th>
        <th>最大误差 (MeV)</th>
        <th>相对改进</th>
    </tr>
    <tr>
        <td><strong>SEMF 基线</strong></td>
        <td>9.57</td>
        <td>5.70</td>
        <td>37.92</td>
        <td>—</td>
    </tr>
    <tr class="failure">
        <td>直接回归</td>
        <td>33.25</td>
        <td>18.94</td>
        <td>92.66</td>
        <td>-247.3%</td>
    </tr>
    <tr class="highlight">
        <td><strong>残差学习 (SEMF+NN)</strong></td>
        <td class="success">3.14</td>
        <td class="success">1.78</td>
        <td class="success">16.79</td>
        <td class="success">+67.2%</td>
    </tr>
    <tr class="failure">
        <td>哈密顿量方法</td>
        <td colspan="4" style="text-align:center">训练失败（架构问题）</td>
    </tr>
</table>

<p class="caption">表1：四种方法在AME2020测试集上的性能对比</p>

<h3>3.2 关键发现</h3>

<p><strong>发现1：物理先验至关重要</strong></p>
<p>
残差学习方法（SEMF+NN）取得了最佳性能，RMSE从9.57 MeV降至3.14 MeV，
相对改进67.2%。这证明在小样本条件下，物理知识作为归纳偏置能够显著提升模型泛化能力。
</p>

<p><strong>发现2：纯数据驱动方法在小样本下失效</strong></p>
<p>
直接回归方法完全失败（RMSE 33.25 MeV，比SEMF差247%），说明在仅有162个训练样本的情况下，
神经网络无法从头学习核结合能的复杂物理规律。这与计算机视觉等大数据领域的成功经验形成鲜明对比。
</p>

<p><strong>发现3：架构设计必须匹配任务特点</strong></p>
<p>
哈密顿量方法虽然理论上更加优雅（直接建模量子力学），但因架构设计问题（O(N⁴)复杂度、
尺度不匹配）未能成功。这提醒我们：物理启发的架构不一定总是最优选择，
必须考虑计算可行性和任务匹配度。
</p>

<h3>3.3 训练效率</h3>

<table>
    <tr>
        <th>模型</th>
        <th>参数量</th>
        <th>训练时间</th>
        <th>收敛轮数</th>
    </tr>
    <tr>
        <td>直接回归</td>
        <td>121,729</td>
        <td>55.6秒</td>
        <td>500 epochs</td>
    </tr>
    <tr class="highlight">
        <td><strong>残差学习</strong></td>
        <td>27,905</td>
        <td>49.5秒</td>
        <td>500 epochs</td>
    </tr>
</table>

<p class="caption">表2：模型训练效率对比（NVIDIA GPU）</p>

<p>
残差学习方法不仅性能最优，而且参数量更少（仅为直接回归的23%），训练速度更快。
这进一步证明了物理先验的价值：通过SEMF提供正确的归纳偏置，神经网络只需学习小的修正项，
因此需要更少的参数和训练时间。
</p>

<h2 class="section-break">4. 讨论</h2>

<h3>4.1 为什么残差学习有效？</h3>

<p>残差学习方法的成功可以从以下角度理解：</p>

<ol>
    <li><strong>正确的标度</strong>：SEMF提供了结合能随质量数A的正确标度关系（~15A MeV），
    神经网络只需学习相对较小的修正（~几MeV）</li>

    <li><strong>物理约束</strong>：SEMF隐含了核力的基本性质（短程、饱和性），
    为神经网络提供了强先验</li>

    <li><strong>降低学习难度</strong>：原问题是学习B(Z,N)（范围0-400 MeV），
    残差学习将其转化为学习ΔB(Z,N)（范围±20 MeV），大大降低了学习难度</li>

    <li><strong>更好的泛化</strong>：SEMF在整个核图上都有合理的外推性能，
    即使在训练集之外的区域，残差修正也能基于SEMF的合理基线进行</li>
</ol>

<h3>4.2 与相关工作的比较</h3>

<p>
近年来多项研究探索了机器学习在核质量预测中的应用。本研究的独特贡献在于：
</p>

<ul>
    <li>系统对比了不同物理先验强度的方法（无先验、强先验、中等先验）</li>
    <li>在小样本条件下（243个核素）验证了物理先验的关键作用</li>
    <li>提供了失败案例分析（直接回归、哈密顿量方法），为后续研究提供经验</li>
</ul>

<h3>4.3 局限性与未来工作</h3>

<p>本研究存在以下局限：</p>

<ul>
    <li><strong>数据范围</strong>：仅限于轻核到中等质量核（Z≤20），重核区域的表现有待验证</li>
    <li><strong>不确定性量化</strong>：当前模型未提供预测不确定度，这对实际应用很重要</li>
    <li><strong>可解释性</strong>：神经网络学到的修正项缺乏明确的物理解释</li>
</ul>

<p>未来可以从以下方向改进：</p>

<ol>
    <li>扩展到全核图（包括重核和超重核）</li>
    <li>引入贝叶斯神经网络进行不确定性量化</li>
    <li>结合注意力机制分析模型关注的物理特征</li>
    <li>探索其他物理量的预测（如分离能、半衰期等）</li>
</ol>

<h2 class="section-break">5. 结论</h2>

<div class="conclusion">
<p>
本研究基于AME2020实验数据，系统比较了四种核结合能预测方法。主要结论如下：
</p>

<ol>
    <li><strong>残差学习方法（SEMF + 神经网络修正）取得最佳性能</strong>，
    在测试集上达到3.14 MeV RMSE，相比SEMF基线（9.57 MeV）改进67.2%</li>

    <li><strong>物理先验在小样本学习中至关重要</strong>。
    抛弃物理知识的直接回归方法完全失败（RMSE 33.25 MeV），
    证明在有限数据下纯数据驱动方法不可行</li>

    <li><strong>架构设计必须平衡物理优雅性与计算可行性</strong>。
    哈密顿量方法虽然理论上更加严格，但因架构问题未能成功</li>

    <li><strong>残差学习提供了一个通用范式</strong>：
    在小样本物理问题中，应优先考虑"物理模型 + 数据驱动修正"的混合方法，
    而非完全抛弃物理知识</li>
</ol>

<p>
本研究为核物理中的机器学习应用提供了重要启示：
<strong>在数据稀缺的科学领域，物理知识不是可选项，而是必需品。</strong>
成功的机器学习模型应该是物理洞察与数据驱动方法的有机结合。
</p>
</div>

<h2 class="section-break">参考文献</h2>

<ol style="font-size: 10pt;">
    <li>Wang, M., et al. (2021). The AME2020 atomic mass evaluation.
    <em>Chinese Physics C</em>, 45(3), 030003.</li>

    <li>Weizsäcker, C. F. von (1935). Zur Theorie der Kernmassen.
    <em>Zeitschrift für Physik</em>, 96(7-8), 431-458.</li>

    <li>Utama, R., Piekarewicz, J., & Prosper, H. B. (2016).
    Nuclear mass predictions for the crustal composition of neutron stars:
    A Bayesian neural network approach. <em>Physical Review C</em>, 93(1), 014311.</li>

    <li>Niu, Z. M., & Liang, H. Z. (2018). Nuclear mass predictions based on
    Bayesian neural network approach with pairing and shell effects.
    <em>Physics Letters B</em>, 778, 48-53.</li>

    <li>Neufcourt, L., et al. (2018). Bayesian approach to model-based extrapolation
    of nuclear observables. <em>Physical Review C</em>, 98(3), 034318.</li>
</ol>

<h2 class="section-break">附录：技术细节</h2>

<h3>A. 模型架构</h3>

<p><strong>残差学习网络结构：</strong></p>
<ul>
    <li>输入层：17个物理启发特征（Z, N, A, A<sup>2/3</sup>, Z²/A<sup>1/3</sup>,
    (N-Z)²/A, 配对项, 幻数距离等）</li>
    <li>隐藏层：128 → 128 → 64 神经元，使用SiLU激活函数和LayerNorm</li>
    <li>Dropout：0.1（防止过拟合）</li>
    <li>输出层：1个神经元（残差修正值）</li>
    <li>最终预测：B<sub>pred</sub> = B<sub>SEMF</sub> + NN<sub>output</sub></li>
</ul>

<h3>B. 训练配置</h3>

<ul>
    <li>优化器：Adam (lr=1e-3, weight_decay=1e-4)</li>
    <li>学习率调度：Cosine annealing (T_max=500, η_min=1e-6)</li>
    <li>批大小：32</li>
    <li>训练轮数：500 epochs</li>
    <li>损失函数：均方误差（MSE）</li>
    <li>硬件：NVIDIA GPU (CUDA 11.8)</li>
</ul>

<h3>C. 数据处理</h3>

<ul>
    <li>数据来源：AME2020质量表（mass_1.mas20.txt）</li>
    <li>筛选条件：Z=2-20, N=2-28, 实验不确定度 < 1 MeV</li>
    <li>数据增强：无（保持数据真实性）</li>
    <li>特征标准化：无（物理特征已具有合理尺度）</li>
</ul>

<div style="margin-top: 2cm; text-align: center; color: #666; font-size: 10pt;">
    <p>—— 报告完 ——</p>
    <p>项目代码：/share/liuyutian/rydberg-rl-entanglement/nuclear-ml</p>
</div>

</body>
</html>
"""

# 生成PDF
print("正在生成PDF报告...")
HTML(string=html_content).write_pdf(
    'nuclear_ml_report_final.pdf',
    stylesheets=[CSS(string='@page { size: A4; margin: 2.5cm; }')]
)

print("✓ PDF报告已生成: nuclear_ml_report_final.pdf")
print(f"  文件大小: {Path('nuclear_ml_report_final.pdf').stat().st_size / 1024:.1f} KB")
print(f"  位置: /share/liuyutian/rydberg-rl-entanglement/nuclear-ml/nuclear_ml_report_final.pdf")

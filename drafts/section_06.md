# 6 AI 控制：PPO + Domain Randomization

本节将量子控制问题重新表述为 Markov 决策过程（MDP），并利用近端策略优化（proximal policy optimization, PPO）算法求解。结合 domain randomization 技术，所训练的神经网络策略不仅能产生高保真度的控制脉冲，还能对 \S5.2 中定义的全部噪声通道保持鲁棒性。

## 6.1 开放量子系统到 MDP 的映射

强化学习（RL）的核心抽象是 MDP 四元组 $(\mathcal{S}, \mathcal{A}, P, r)$。我们将 \S5 中定义的开放量子动力学系统逐一映射到这一框架中。

**状态空间 $\mathcal{S}$。** 系统状态由密度矩阵 $\rho(t)$ 完整描述。对于双原子四能级系统，$\rho$ 是 $4 \times 4$ 的复 Hermitian 矩阵。利用 Hermiticity（$\rho_{ij} = \rho_{ji}^*$）和迹归一条件（$\text{Tr}\,\rho = 1$），独立实参数为 $4^2 - 1 = 15$ 个。在实际实现中，我们将 $\rho$ 的上三角部分（含对角）展开为实向量：对角元取实部（4 个），上三角非对角元取实部和虚部（$6 \times 2 = 12$ 个），再附加时间 $t/T$ 作为额外特征，得到

$$s_t = \bigl(\text{Re}\,\rho_{11}, \ldots, \text{Re}\,\rho_{44},\; \text{Re}\,\rho_{12}, \text{Im}\,\rho_{12}, \ldots,\; t/T\bigr) \in \mathbb{R}^{17}$$

在简化实现中（不利用对称性约化），也可以将完整的 $4 \times 4$ 复矩阵拆为实部和虚部，得到 32 维实向量——这为后续扩展到更大 Hilbert 空间时保留了通用性。

**动作空间 $\mathcal{A}$。** 每个时间步，智能体输出两个连续控制变量：

$$a_t = \bigl(\Omega(t),\; \Delta(t)\bigr) \in \mathbb{R}^2$$

分别对应 Rabi 频率（激光强度）和失谐（激光频率偏移）。这两个参数通过可编程的声光调制器（AOM）或电光调制器（EOM）在实验中逐时间步可调。动作空间的物理约束为 $0 \leq \Omega(t) \leq \Omega_{\max}$，$|\Delta(t)| \leq \Delta_{\max}$。

**状态转移 $P$。** 系统演化遵循 Lindblad 主方程：

$$\dot\rho = -i[H(t),\,\rho] + \sum_k\!\left(L_k\rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k,\,\rho\}\right) \tag{6.1}$$

其中 $H(t)$ 为含控制参数 $(\Omega(t), \Delta(t))$ 的系统 Hamiltonian（参见 \S3.2 的 Eq.(3.2)），$\{L_k\}$ 为 \S5.2 中定义的 Lindblad 算符。在离散时间步下，每步的时间间隔 $\delta t$ 内，通过四阶 Runge-Kutta 方法数值积分 Eq.(6.1) 完成状态转移 $\rho(t) \to \rho(t+\delta t)$。经典随机噪声（多普勒频移、位置抖动、OU 强度噪声、相位噪声）在每步开始时重新采样并叠加到 Hamiltonian 参数中。

**奖励函数。** 采用稀疏终端奖励（sparse terminal reward）设计：

$$r_t = \begin{cases} \text{Tr}\bigl(\rho(T)\,\rho_{\text{tgt}}\bigr) & \text{if } t = T \\ 0 & \text{otherwise} \end{cases} \tag{6.2}$$

稀疏奖励的物理动机在于：量子态保真度只在门操作结束时才具有明确的物理意义——中间时刻的瞬时保真度并不代表最终的控制质量（中间态可能经历远离目标态的复杂路径后最终到达高保真度终态）。这与经典控制中常用的积分代价函数形成对比，也对 RL 算法的信用分配（credit assignment）能力提出了更高要求。

## 6.2 PPO 算法核心

### 6.2.1 策略梯度与 clipped surrogate 目标

PPO [PPO17] 是一种 on-policy 策略梯度算法。其核心思想是在每次策略更新中限制新旧策略之间的偏移幅度，从而在样本效率和训练稳定性之间取得平衡。

定义概率比：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

其中 $\pi_\theta$ 为参数化策略网络（输出高斯分布的均值和方差），$\hat{A}_t$ 为 generalized advantage estimation (GAE) 计算的优势函数。PPO 的 clipped surrogate 目标函数为

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\text{clip}\bigl(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\bigr)\hat{A}_t\right)\right] \tag{6.3}$$

其中 $\epsilon = 0.2$ 为 clipping 超参数。该目标函数的几何意义如下：

- 当 $\hat{A}_t > 0$（当前动作优于平均）时，$L^{\text{CLIP}}$ 鼓励增大 $r_t(\theta)$（使该动作更可能被选取），但通过 $\text{clip}$ 将增幅限制在 $1+\epsilon$ 以内；
- 当 $\hat{A}_t < 0$（当前动作劣于平均）时，$L^{\text{CLIP}}$ 鼓励减小 $r_t(\theta)$，但下界被限制在 $1-\epsilon$。

完整的训练目标还包括价值函数损失 $L^V$ 和策略熵正则项 $H[\pi_\theta]$：

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1\,L^V(\theta) + c_2\,H[\pi_\theta] \tag{6.4}$$

其中 $c_1 = 0.5$（价值函数系数），$c_2 = 0.01$（熵系数，防止策略过早收敛到局部最优）。

### 6.2.2 为何选择 PPO 而非 DDPG/SAC

在连续控制任务中，off-policy 算法（如 DDPG、TD3、SAC）通常具有更高的样本效率。然而，量子控制问题的特殊结构使得 PPO（on-policy）成为更优选择，原因如下：

**量子保真度景观的 spin-glass 特性。** Bukov *et al.* [Bukov18] 在 PRX 2018 中系统研究了量子控制景观的拓扑结构，发现其呈现类似自旋玻璃（spin-glass）的多极值特征：大量近乎等深的局部极值被高能垒分隔。在这种景观中，off-policy 算法的 experience replay buffer 存储了来自旧策略的样本，经过 Q 值的 bootstrap 估计后，误差会在这些局部极值间传播和放大，导致训练不稳定甚至发散。

**直接的数值证据。** Ernst *et al.* [Ernst25]（ICML 2025，arXiv:2501.14372）在 Rydberg CZ 门控制任务上系统比较了 PPO、DDPG 和 TD3 三种算法。结果表明：(i) PPO 的最终保真度比 DDPG/TD3 高出约一个数量级（infidelity 降低 10 倍以上）；(ii) PPO 的训练曲线更平滑，收敛更可靠；(iii) DDPG 和 TD3 频繁陷入局部极值，且对超参数极为敏感。

这些发现与上述 spin-glass 景观分析一致：on-policy 算法避免了 replay buffer 中过期样本的干扰，且 PPO 的 clipping 机制天然限制了策略更新幅度，防止在复杂景观中的灾难性跳跃。

## 6.3 Domain Randomization 的物理意义

### 6.3.1 训练范式

Domain randomization [Niu19] 的核心思想极为直截：在每个训练 episode 开始时，从 \S5.2 中定义的分布重新采样所有噪声参数——

- 多普勒频移 $\delta_i \sim \mathcal{N}(0, \sigma_D^2)$
- 位置抖动 $\delta R \sim \mathcal{N}(0, \sigma_R^2)$
- OU 噪声初始条件 $\xi_0 \sim \mathcal{N}(0, \sigma^2/2\theta)$
- 相位噪声实例
- Lindblad 衰减速率（可在 $\pm 10\%$ 范围内抖动以模拟 $\tau_{\text{eff}}$ 的不确定性）

然后在该组噪声实例下完成一个完整的 episode。策略网络在数以万计的、各不相同的噪声环境中反复训练，其参数 $\theta$ 被优化为在整个噪声分布上的期望保真度最大化：

$$\theta^* = \arg\max_\theta\;\mathbb{E}_{\xi \sim p(\xi)}\!\left[\text{Tr}\bigl(\rho_\xi(T)\,\rho_{\text{tgt}}\bigr)\right] \tag{6.5}$$

### 6.3.2 从单一脉冲到噪声鲁棒策略

传统最优控制方法（如 GRAPE）输出的是一条固定的脉冲序列 $\{\Omega(t_k), \Delta(t_k)\}_{k=1}^N$——它针对**特定的噪声实例**（或无噪声情况）进行优化，是一种开环控制（open-loop control）。当实际噪声偏离假设时，保真度迅速下降。

与之形成鲜明对比的是，domain randomization 训练出的神经网络策略 $\pi_\theta(a_t | s_t)$ 是一个**从当前量子态到控制动作的映射**。由于训练过程中系统状态 $s_t = \rho(t)$ 已经隐含了所有噪声的影响（密度矩阵是噪声作用后的真实状态），策略网络学会了根据当前状态自适应地调整控制参数。这本质上是一种**闭环反馈控制**（closed-loop feedback control），尽管在实验中直接获取 $\rho(t)$ 需要量子态层析（quantum state tomography），但在仿真环境中这一信息是免费的。

### 6.3.3 物理类比：自适应动态解耦

从物理直觉出发，domain randomization 训练出的策略可以类比为一种**自适应的动态解耦序列**（adaptive dynamical decoupling）[Niu19, Guatto24]。传统的动态解耦（如 spin echo、CPMG）通过在固定时刻插入 $\pi$ 脉冲来平均掉慢涨落噪声，但其脉冲时序是预先设计的，无法应对未知的噪声谱。

domain randomization 策略则更进一步：它学会了**根据当前状态自适应地调制相位（chirp）和幅度**，使得系统在整个演化过程中持续与噪声谱解耦。具体而言：

- 当多普勒频移导致失谐偏移时，策略自动调整 $\Delta(t)$ 进行补偿；
- 当激光强度涨落降低有效 Rabi 频率时，策略增大 $\Omega(t)$ 或延长驱动时间；
- 当位置抖动改变 van der Waals 相互作用强度时，策略调整脉冲时序以适配新的阻塞条件。

这种"万能的"噪声应对能力正是 domain randomization 赋予 RL 策略的核心优势。Guatto *et al.* [Guatto24] 在 *Nature Communications* 2024 中展示了类似的 domain randomization 策略在超导量子比特系统中实现了超越最优控制方法的门保真度，验证了这一范式的普适性。

---

**参考文献**

- [Bukov18] A. G. Bukov *et al.*, "Reinforcement learning in different phases of quantum control," *Phys. Rev. X* **8**, 031086 (2018).
- [Niu19] M. Y. Niu *et al.*, "Universal quantum control through deep reinforcement learning," *npj Quantum Inf.* **5**, 33 (2019).
- [PPO17] J. Schulman *et al.*, "Proximal policy optimization algorithms," arXiv:1707.06347 (2017).
- [Ernst25] O. Ernst *et al.*, "Reinforcement learning for Rydberg quantum gates," *ICML 2025*, arXiv:2501.14372 (2025).
- [DingEnglund25] Y. Ding, D. Englund *et al.*, arXiv:2504.11737 (2025).
- [Guatto24] S. Guatto *et al.*, "Model-free quantum gate design and calibration using deep reinforcement learning," *Nat. Commun.* **15**, 8353 (2024).

# Alkox 任务完整技术流水线：从仿真器到 RL 训练

> 本文档以 alkox（4D 化学反应优化）任务为例，详细讲解 CAP-PPO 框架中"从工业仿真器出发，经过变体生成、BNN 拟合、合成函数构造，最终训练 RL 策略"的完整技术流水线。每一步都给出原理和关键公式。

---

## 0. 问题背景与动机

### 0.1 工业场景

我们面对的场景是：一个化学实验室历史上做过**少量实验**（比如 10 组不同配置，每组 30 次观测），希望训练一个智能 BO 策略，使其在**未来遇到类似但不完全相同的新实验**时，能够用极少的交互次数找到最优条件。

关键约束：
- **历史实验有限**：不能无限制地做新实验（成本高、耗时长）
- **不能在线交互**：训练 RL 策略时不能实时调用实验装置
- **需要泛化**：策略要能适用于与历史实验相似但不完全相同的新任务

### 0.2 解决思路

```
有限的历史实验数据
    ↓ BNN 拟合（学习函数的概率模型）
代理模型（BNN 后验均值）
    ↓ + 随机扰动（RFF）生成大量合成函数
无限多样的训练环境
    ↓ RL 训练（PPO）
泛化 BO 策略
```

在我们的框架中，用 Olympus 仿真器 + 仿射变换来**模拟**这个工业场景。

---

## 1. 基础仿真器：Olympus Alkox

### 1.1 仿真器简介

Alkox 是 Olympus 提供的一个确定性神经网络（NeuralNet）仿真器，模拟烷氧化（alkoxylation）化学反应。

- **输入**：4 个连续变量（催化酶浓度、过氧化物酶浓度、醇氧化酶浓度、pH）
- **输出**：转化率（conversion），标量
- **目标**：最大化转化率

**真实物理范围**：

| 变量 | 下界 | 上界 | 单位 |
|------|------|------|------|
| catalase | 0.05 | 1.0 | U/mL |
| peroxidase | 0.5 | 10.0 | U/mL |
| alcohol_oxidase | 2.0 | 8.0 | U/mL |
| ph | 6.0 | 8.0 | — |

### 1.2 归一化

优化器统一在 $[0,1]^4$ 归一化空间上运行。归一化与真实尺度之间的映射：

$$x_{\text{real}} = x_{\text{lower}} + x_{\text{norm}} \cdot (x_{\text{upper}} - x_{\text{lower}})$$

其中 $x_{\text{norm}} \in [0,1]^4$。

### 1.3 目标取反

框架统一采用**最小化**目标。由于 alkox 仿真器的目标是**最大化**转化率，因此取反：

$$f(x) = -\text{emulator}(x_{\text{real}})$$

即最小化负转化率 = 最大化转化率。

---

## 2. 第一步：仿射输入变换 — 从仿真器生成变体

### 2.1 动机

一个仿真器只提供一个固定的目标函数。但训练 RL 策略需要一族**相似但不完全相同**的函数（模拟不同实验室/不同批次的实验）。我们通过对仿真器的**输入空间施加仿射变换**来生成变体。

### 2.2 仿射变换公式

给定归一化输入 $\mathbf{x} \in [0,1]^4$，变换后的输入为：

$$\mathbf{x}' = \text{clip}\Big(\mathbf{c} + \mathbf{R} \cdot \mathbf{S} \cdot (\mathbf{x} - \mathbf{c}) + \mathbf{d},\ 0,\ 1\Big)$$

其中：

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbf{c} = (0.5, 0.5, 0.5, 0.5)$ | 变换中心 | $[0,1]^4$ 的中心点 |
| $\mathbf{S} = \text{diag}(s_1, s_2, s_3, s_4)$ | 缩放矩阵 | 每维独立缩放 |
| $\mathbf{R}$ | 旋转矩阵 | $4 \times 4$ 正交矩阵 |
| $\mathbf{d} = (d_1, d_2, d_3, d_4)$ | 平移向量 | 整体位移 |
| $\text{clip}(\cdot, 0, 1)$ | 逐元素裁切 | 确保输出仍在 $[0,1]^4$ 内 |

变换后调用仿真器：

$$f_{\text{variant}}(\mathbf{x}) = -\text{emulator}\big(\text{real\_scale}(\mathbf{x}')\big)$$

### 2.3 旋转矩阵的构造（Givens 旋转）

4D 空间中，旋转矩阵由 4 个 **Givens 旋转** 复合而成。每个 Givens 旋转在一个 2D 平面内旋转：

$$G_{ij}(\theta) = \begin{pmatrix}
\ddots & & & \\
& \cos\theta & \cdots & -\sin\theta & \\
& \vdots & \ddots & \vdots & \\
& \sin\theta & \cdots & \cos\theta & \\
& & & & \ddots
\end{pmatrix}$$

选择的 4 个旋转平面为 $(0,1)$, $(2,3)$, $(0,2)$, $(1,3)$，**每个维度恰好参与 2 个平面**，确保变换均匀覆盖所有方向。

$$\mathbf{R} = G_{01}(\theta_1) \cdot G_{23}(\theta_2) \cdot G_{02}(\theta_3) \cdot G_{13}(\theta_4)$$

### 2.4 变体参数空间

每个变体由 12 个参数定义：

| 参数 | 数量 | 作用 | 采样范围（in_range） |
|------|------|------|---------------------|
| $d_1, d_2, d_3, d_4$ | 4 | 平移 | $\text{Uniform}(-0.08, +0.08)$ |
| $s_1, s_2, s_3, s_4$ | 4 | 缩放 | $\text{Uniform}(0.84, 1.00)$ |
| $\theta_{01}, \theta_{23}, \theta_{02}, \theta_{13}$ | 4 | 旋转角（度） | $\text{Uniform}(-28°, +28°)$ |

设计要点：
- **缩放仅缩小（$s \le 1$）**：避免放大后超出 $[0,1]^4$ 边界被大面积裁切
- **平移幅度适中（$\pm 0.08$）**：足以移动最优点位置，又不会使变换区域过多落在边界外
- **旋转角适中（$\pm 28°$）**：足以改变函数景观的方向性

### 2.5 直觉理解

仿射变换的效果：

- **平移 $\mathbf{d}$**：整体移动函数的最优点位置
- **缩放 $\mathbf{S}$**：压缩搜索空间，使函数变得"更窄"或"更宽"
- **旋转 $\mathbf{R}$**：改变各维度之间的相关结构

这模拟了**不同实验批次之间的系统性偏差**——函数形状相似，但最优条件和梯度方向不同。

### 2.6 采样 10 个训练变体

```python
variants = task.sample_train_variants(k=10, seed=2026)
# 返回 10 个 dict，每个包含 dx1..dx4, sx1..sx4, r01, r23, r02, r13
```

这 10 个变体模拟了"10 组历史实验"。

---

## 3. 第二步：BO 轨迹采集 — 模拟历史实验数据

### 3.1 动机

在工业场景中，每组历史实验产生了有限的观测数据（输入-输出对）。我们通过在每个变体上运行标准 GP+EI 贝叶斯优化来**模拟**这个数据采集过程。

### 3.2 GP + EI 优化过程

对每个变体 $i$（$i = 0, \ldots, 9$），运行 GP+EI BO：

1. **随机初始化**：在 $[0,1]^4$ 中均匀随机采样 $n_{\text{init}} = 2$ 个点
2. **迭代优化**（共 $T = 28$ 步）：
   - 用已有数据 $\{(\mathbf{x}_j, y_j)\}_{j=1}^{t}$ 拟合 GP（Matern 2.5 核）
   - 计算 EI（Expected Improvement）采集函数：
     $$\text{EI}(\mathbf{x}) = \mathbb{E}\big[\max(f^* - Y(\mathbf{x}),\ 0)\big]$$
     其中 $f^* = \min_{j \le t} y_j$，$Y(\mathbf{x}) \sim \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$ 为 GP 后验
   - 选择 $\mathbf{x}_{t+1} = \arg\max_\mathbf{x} \text{EI}(\mathbf{x})$
   - 评估 $y_{t+1} = f_{\text{variant}_i}(\mathbf{x}_{t+1})$
3. **总计**：每个变体 $2 + 28 = 30$ 个观测点

### 3.3 Best-of-5 试验选择

由于 GP+EI 受初始随机点影响较大，对每个变体独立运行 5 次，**保留最优的一次**（最终 best $y$ 最小的那次）。这模拟了实验人员会选择表现最好的实验记录来作为参考数据。

### 3.4 输出数据格式

```
X_trajs: (10, 30, 4)    # 10 个变体，每个 30 个点，4 维输入
y_trajs: (10, 30)        # 对应的函数值
variant_indices: (10,)    # 变体编号
```

**这就是我们的"历史实验数据"**——每个变体 30 个 $(x, y)$ 对。

---

## 4. 第三步：训练 BNN — 从有限数据学习代理模型

### 4.1 动机

有了 10 组历史实验数据（每组 30 个点），我们需要从中学习一个**概率代理模型**。选择贝叶斯神经网络（BNN）是因为：

- NN 函数类天然匹配 Olympus 的 NN 仿真器（GP 在此类景观上拟合不稳定）
- 变分后验提供不确定性估计
- 训练配置与 Olympus 内置的 BayesNeuralNet 完全对齐，确保一致性

### 4.2 BNN 架构

对每个变体 $i$ 独立训练一个 BNN：

$$\text{BNN}_i: \mathbb{R}^4 \to \mathbb{R}$$

网络结构：

```
Input (4) → Dense(48) → LeakyReLU → Dense(48) → LeakyReLU → Dense(48) → LeakyReLU → Dense(1)
```

- **隐藏层**：3 层，每层 48 个节点
- **激活函数**：LeakyReLU（$\alpha = 0.2$）

$$\text{LeakyReLU}(h) = \begin{cases} h & \text{if } h > 0 \\ 0.2 \cdot h & \text{otherwise} \end{cases}$$

### 4.3 变分推断

每一层的权重不是固定值，而是**高斯变分后验**：

$$w_{ij} \sim q(w_{ij}) = \mathcal{N}(\mu_{ij},\ \sigma_{ij}^2)$$

使用 TensorFlow Probability 的 `DenseLocalReparameterization` 层实现，通过重参数化技巧高效采样：

$$w_{ij} = \mu_{ij} + \sigma_{ij} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

训练结束后，每一层保存两组参数：
- $\mu$（`loc`）：后验均值 — 最佳权重估计
- $\sigma$（`sigma`）：后验标准差 — 不确定性宽度

### 4.4 训练目标（ELBO）

训练损失函数为负 ELBO（Evidence Lower Bound）：

$$\mathcal{L} = \underbrace{-\sum_{n=1}^{B} \log p(y_n \mid \mathbf{x}_n, \mathbf{w})}_{\text{NLL（数据拟合项）}} + \lambda \cdot \underbrace{\frac{1}{B} \text{KL}\big[q(\mathbf{w}) \| p(\mathbf{w})\big]}_{\text{KL 正则项（除以 batch size）}}$$

其中：

#### 数据拟合项（NLL）

$$-\log p(y_n \mid \mathbf{x}_n, \mathbf{w}) = -\log \mathcal{N}(y_n;\ \hat{y}_n,\ \sigma_{\text{noise}}^2)$$

- $\hat{y}_n$ 是 BNN 前向传播的预测值
- $\sigma_{\text{noise}} = \text{softplus}(\gamma)$ 是可学习的异方差噪声尺度（aleatoric uncertainty）
- 使用 `reduce_sum` 而非 `reduce_mean`（Olympus 惯例）

#### KL 正则项

$$\text{KL}\big[q(\mathbf{w}) \| p(\mathbf{w})\big] = \sum_{\text{layers}} \text{KL}\big[\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)\big]$$

- 先验 $p(\mathbf{w}) = \mathcal{N}(0, \mathbf{I})$
- KL 除以 batch size $B$（Olympus 惯例）
- $\lambda = 0.001$ 为 KL 权重（alkox 配置）

$\lambda$ 的作用：
- $\lambda$ 越小 → 后验越紧（$\sigma$ 小）→ BNN 越精确但多样性越低
- $\lambda$ 越大 → 后验越宽 → 多样性高但拟合质量下降

### 4.5 训练流程

| 配置 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam, lr=1e-3 | — |
| batch_size | 20 | 有放回随机采样（Olympus 惯例） |
| max_epochs | 100,000 | — |
| 早停 | patience=100 | 每 100 epoch 检查验证集 RMSD |
| 验证集比例 | 20% | 从 30 个点中分出 6 个 |
| Y 标准化 | $\hat{y} = (y - \bar{y}) / \sigma_y$ | 预处理 |

### 4.6 训练输出

对每个变体 $i$，保存：

```
layer_i_0_loc, layer_i_0_sigma, layer_i_0_bias    # 第 1 层: (4, 48), (4, 48), (48,)
layer_i_1_loc, layer_i_1_sigma, layer_i_1_bias    # 第 2 层: (48, 48), ...
layer_i_2_loc, layer_i_2_sigma, layer_i_2_bias    # 第 3 层: (48, 48), ...
layer_i_3_loc, layer_i_3_sigma, layer_i_3_bias    # 输出层: (48, 1), ...
y_mean_i, y_std_i                                  # Y 标准化参数
```

10 个变体 → 10 组独立的 BNN 参数。

---

## 5. 第四步：BNN 后验均值 + RFF 扰动 — 构造合成训练函数

### 5.1 动机

训练 RL 策略需要**大量多样的**目标函数。但我们只有 10 个 BNN（对应 10 个变体），直接使用它们的后验均值只有 10 个固定函数，远远不够。

需要解决的核心问题：**如何从 10 个 BNN 生成无限多样的合成函数？**

### 5.2 曾尝试但失败的方案

**方案 A：BNN 权重采样**
$$w = \mu + |\sigma| \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

问题：Olympus-aligned 训练使后验非常紧（$\sigma \ll |\mu|$），采样的函数几乎与均值相同（inter-sample 相关性 > 0.89）。

**方案 B：增大 KL 权重**

放大 $\lambda$ 到 0.003 可以增加多样性（inter 降到 0.53），但同时**牺牲了 BNN 的拟合质量**。

**方案 C：温度放大**
$$w = \mu + T \cdot |\sigma| \cdot \epsilon, \quad T > 1$$

温度 $T > 1$ 直接导致 NN 权重扰动过大，前向传播产生极端值，函数退化为噪声。

### 5.3 最终方案：BNN_mean + RFF 输出扰动

核心思想：**将拟合质量和多样性完全解耦**。

$$\boxed{f_{\text{synth}}(\mathbf{x}) = \underbrace{f_{\text{BNN}}^{\mu}(\mathbf{x})}_{\text{准确的基函数}} + \alpha \cdot \underbrace{g(\mathbf{x})}_{\text{平滑随机扰动}}}$$

- $f_{\text{BNN}}^{\mu}$：BNN 后验均值前向传播（仅使用 $\mu$，不使用 $\sigma$）
- $g(\mathbf{x})$：从 GP 先验采样的随机函数（Random Fourier Feature）
- $\alpha = 5.0$：扰动强度

关键区别：$g(\mathbf{x})$ 是从 GP **先验**（而非后验）采样的，**不拟合任何数据**，仅提供平滑的随机扰动。

### 5.4 BNN 后验均值前向传播

给定 BNN 的 $\mu$ 权重（$L$ 层），前向传播为：

$$\mathbf{h}^{(0)} = \mathbf{x}$$

$$\mathbf{h}^{(l)} = \text{LeakyReLU}\big(\mathbf{h}^{(l-1)} \mathbf{W}^{(l)}_\mu + \mathbf{b}^{(l)}\big), \quad l = 1, \ldots, L-1$$

$$\hat{y}_{\text{norm}} = \mathbf{h}^{(L-1)} \mathbf{W}^{(L)}_\mu + \mathbf{b}^{(L)}$$

$$f_{\text{BNN}}^{\mu}(\mathbf{x}) = \bar{y} + \sigma_y \cdot \hat{y}_{\text{norm}}$$

其中 $\bar{y}, \sigma_y$ 是训练时保存的 Y 标准化参数。

### 5.5 Random Fourier Features（RFF）— 从 GP 先验采样平滑函数

#### 5.5.1 理论基础：Bochner 定理

对于一个平稳核 $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x} - \mathbf{x}')$，Bochner 定理保证：

$$k(\boldsymbol{\tau}) = \int e^{i \boldsymbol{\omega}^T \boldsymbol{\tau}} \, p(\boldsymbol{\omega}) \, d\boldsymbol{\omega}$$

其中 $p(\boldsymbol{\omega})$ 是核的**谱密度**（spectral density）。

由此可知，从 GP 先验 $\mathcal{GP}(0, k)$ 中采样的函数可以用有限个随机傅里叶特征近似：

$$g(\mathbf{x}) \approx \sqrt{\frac{2}{M}} \sum_{m=1}^{M} \cos(\boldsymbol{\omega}_m^T \mathbf{x} + b_m)$$

其中 $\boldsymbol{\omega}_m \sim p(\boldsymbol{\omega})$，$b_m \sim \text{Uniform}(0, 2\pi)$，$M$ 为特征数量。

#### 5.5.2 Matern 2.5 核的谱密度

我们使用 Matern $\nu = 5/2$ 核：

$$k(\boldsymbol{\tau}) = \sigma^2 \left(1 + \frac{\sqrt{5}\|\boldsymbol{\tau}\|}{l} + \frac{5\|\boldsymbol{\tau}\|^2}{3l^2}\right) \exp\left(-\frac{\sqrt{5}\|\boldsymbol{\tau}\|}{l}\right)$$

其谱密度为 scaled Student-t 分布。实际采样中，利用等价关系：

$$\boldsymbol{\omega} = \frac{\mathbf{z}}{l \cdot \sqrt{v / (2\nu)}}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d), \quad v \sim \chi^2(2\nu)$$

即先从标准正态采样 $\mathbf{z}$，再从卡方分布 $\chi^2(2\nu = 5)$ 采样 $v$，然后除以长度尺度 $l$ 并按 $v$ 缩放。

#### 5.5.3 具体实现

```python
# 采样 M=256 个随机频率
z ~ N(0, I)          # shape: (256, 4)
v ~ chi2(df=5)       # shape: (256, 1)
W = z / (l * sqrt(v / 5))  # shape: (256, 4)

# 采样 M 个随机相位
b ~ Uniform(0, 2*pi)  # shape: (256,)

# 评估
g(x) = sqrt(2/256) * sum_m cos(W_m^T x + b_m)
```

**不同的随机种子 → 不同的 $(W, b)$ → 不同的 $g(\mathbf{x})$ → 不同的合成函数**。

#### 5.5.4 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| $M$ | 256 | RFF 特征数量（精度与计算的平衡） |
| $l$ | 0.3 | 长度尺度（在 $[0,1]^4$ 域中覆盖约 30%，控制扰动的平滑程度） |
| $\alpha$ | 5.0 | 扰动强度（控制合成函数偏离 BNN 均值的幅度） |

$l$ 的直觉：
- $l$ 大 → $g(\mathbf{x})$ 变化缓慢 → 扰动是大尺度的平移
- $l$ 小 → $g(\mathbf{x})$ 变化频繁 → 扰动是局部的起伏

### 5.6 合成函数的性质

每个合成函数 $f_{\text{synth}}(\mathbf{x}) = f_{\text{BNN}}^{\mu}(\mathbf{x}) + \alpha \cdot g(\mathbf{x})$ 具有以下性质：

1. **确定性**：给定 RFF 种子，$g(\mathbf{x})$ 是固定的 → $f_{\text{synth}}$ 是确定性函数
2. **平滑性**：$g(\mathbf{x})$ 由 Matern 2.5 核保证足够平滑
3. **保真性**：基函数 $f_{\text{BNN}}^{\mu}$ 是对真实函数的最佳估计
4. **多样性**：不同 RFF 种子产生不同的最优点位置和函数值

### 5.7 测试结果

10 个 BNN × 10 个 RFF 种子 = 100 个合成函数：

| 指标 | 值 | 含义 |
|------|-----|------|
| inter-sample 相关性 | **0.156** | 函数间足够不同（越低越好） |
| $f^*$ 标准差 | **22.4** | 最优值变化显著 |

对比 Hartmann 6D 成功训练时的 oracle_gp 指标（inter=0.193），本方案的多样性更好。

---

## 6. 第五步：RL 训练 — 用合成函数训练 CAP-PPO 策略

### 6.1 训练循环

每个 episode 的执行流程：

```
1. 随机选择一个变体 i ~ Uniform{0, ..., 9}
2. 取变体 i 的 BNN 后验均值
3. 生成一个新的 RFF（随机种子）
4. 组合得到合成函数: f(x) = BNN_mean_i(x) + 5.0 * RFF(x)
5. Sobol 探测: 估计 f 的 global_min 和 y_range（用于 reward 归一化）
6. 随机初始化 2 个点，获得初始观测
7. Agent 执行 28 步 BO（每步选择一个候选点）
8. 计算累积 reward，更新 PPO 策略
```

关键点：**每个 episode 的合成函数都是不同的**（因为 RFF 种子不同），但都保持了与真实仿真器相似的函数形状（因为 BNN mean 是准确的基函数）。

### 6.2 训练规模

| 参数 | 值 |
|------|-----|
| 总 episodes | 5000 |
| 每 episode 步数 | 28（+ 2 初始点 = 30 总评估） |
| 每步候选点数 | 192（128 Sobol 基底 + 64 自适应局部） |
| PPO 更新频率 | 每 20 episodes |

### 6.3 关于收敛

参考 Hartmann 6D 的训练日志：约 3000 episodes 后 avg_regret 趋于平稳，5000 episodes 是一个合理的训练预算。

---

## 7. 完整流水线总览

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: 仿射变换生成变体                                        │
│                                                                  │
│  Olympus alkox 仿真器 (4D, NN)                                   │
│       ↓  × 10 组仿射变换 (平移+旋转+缩放)                         │
│  10 个变体函数 (模拟 10 组历史实验)                                │
├──────────────────────────────────────────────────────────────────┤
│  Step 2: GP+EI BO 采集轨迹数据                                   │
│                                                                  │
│  每个变体 → 5 次独立 GP+EI BO → 选最优的 1 次                     │
│       ↓                                                          │
│  10 × 30 个 (x, y) 观测数据 (模拟历史实验记录)                    │
├──────────────────────────────────────────────────────────────────┤
│  Step 3: BNN 拟合                                                │
│                                                                  │
│  对每个变体独立训练一个 BNN (48×3, LeakyReLU, Olympus-aligned)     │
│       ↓                                                          │
│  10 个 BNN 后验 → 保存 (μ, σ, bias) + (y_mean, y_std)           │
├──────────────────────────────────────────────────────────────────┤
│  Step 4: 合成训练函数                                             │
│                                                                  │
│  f(x) = BNN_mean_i(x) + α · RFF(x; seed)                        │
│                                                                  │
│  每个 episode: 随机选变体 i, 随机 RFF seed → 新的确定性函数         │
│  10 个 BNN × ∞ 个 RFF seed = ∞ 个多样的训练函数                   │
├──────────────────────────────────────────────────────────────────┤
│  Step 5: PPO 训练 (5000 episodes)                                │
│                                                                  │
│  Agent 在每个合成函数上执行 BO (2 init + 28 steps = 30 evals)      │
│  优化目标: 最大化累积 improvement                                  │
│       ↓                                                          │
│  训练好的 CAP-PPO 策略                                            │
└──────────────────────────────────────────────────────────────────┘
```

### 对应代码/脚本

| 步骤 | 脚本/函数 | 输出文件 |
|------|----------|---------|
| Step 1+2 | `MYRL/scripts/finetune.py --stage generate` | `data/alkox_emulator_variants_k10_*.npz`<br>`data/alkox_emulator_bo_trajs_k10_*.npz` |
| TAF 数据 | `myrl.rl.train_rl.prepare_taf_data()` | `data/taf_source_data_alkox_*.pkl` |
| Step 3 | `MYRL/scripts/train_bnn_surrogates.py` | `data/bnn_surrogates_alkox_*.npz` |
| Step 4+5 | `MYRL/scripts/train_rl.py --objective_source bnn` | `runs/ppo_alkox_*/ppo_final.pt` |

### 运行命令

```bash
bash MYRL/local_scripts/alkox.sh
```

---

## 附录 A：为什么不用 GP 拟合 Alkox

GP（Matern 2.5 核）在 Alkox 的 NN 仿真器上拟合严重失败：

| 指标 | Hartmann 6D (GP OK) | Alkox (GP FAIL) |
|------|---------------------|-----------------|
| length_scale 正常比例 | 9/10 | **1/10** |
| GP Spearman vs true | 0.518 | **~0.16** |
| GP → RFF 质量 | 有效 | **纯噪声** |

原因：GP Matern 2.5 假设函数是平稳且特定光滑性的，而 NN 仿真器的景观是非平稳的、具有 NN 特有的分段线性结构。两者函数类不匹配。

## 附录 B：RFF 与 oracle_gp 方案的区别

| | oracle_gp (GP→RFF) | BNN_mean + RFF |
|---|---|---|
| GP 的角色 | **拟合**轨迹数据 → 学后验 | 无 GP |
| RFF 的角色 | 从 GP **后验**采样函数 | 从 GP **先验**采样扰动 |
| 失败模式 | GP 拟合不好 → RFF 为噪声 | BNN mean 不好 → 基函数偏差 |
| 对 NN 景观 | 不适用（函数类不匹配） | 适用（BNN 的 NN 函数类匹配） |

---

*生成时间：2026-03-19*

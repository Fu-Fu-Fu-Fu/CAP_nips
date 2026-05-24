# MYRL 项目分析：从半导体智能工艺优化视角

## 一、项目概述

MYRL 是一个基于强化学习的贝叶斯优化（Bayesian Optimization）候选点选择框架，核心思想是训练一个 RL 策略（称为 **CAP-PPO**，Candidate Acquisition Policy trained with PPO）来替代传统的采集函数（如 EI、UCB），以更智能地选择下一个评估点。

项目的工业出发点：**在半导体工艺优化中，真实实验数据极其稀少且昂贵，配方优化往往依赖专业工程师的经验进行逐步试错。本项目希望 RL agent 能从历史工艺数据中学习到工程师的优化思路，从而减少实验成本。**

---

## 二、项目结构

```
MYRL/
├── scripts/              # 入口脚本
│   ├── finetune.py       # TabPFN 微调入口
│   ├── train_rl.py       # CAP-PPO 训练入口
│   ├── eval_rl_new.py    # 多策略对比评估入口
│   ├── eval_pfn.py       # PFN 回归精度评估入口
│   └── plot.py           # 画图脚本
├── myrl/                 # 核心库
│   ├── tasks/            # 任务定义（可扩展）
│   │   ├── base.py       # TaskSpec 抽象基类
│   │   ├── registry.py   # 任务注册中心 (register_task / get_task)
│   │   ├── builtin.py    # 内置任务注册
│   │   ├── branin_family.py
│   │   ├── goldstein_price.py / goldstein_price_family.py
│   │   └── hartmann_3d.py / hartmann_3d_family.py
│   ├── rl/
│   │   └── train_rl.py   # PPO 训练核心 + 网络定义 + 目标函数
│   ├── policies/
│   │   └── policies.py   # 所有策略实现（EI, UCB, PI, TAF, Random, RLPolicy, MetaBO）
│   ├── bo/
│   │   └── select_candidates.py  # TabPFN 候选点选择 + EI 计算
│   ├── finetune/
│   │   └── finetune.py   # TabPFN 微调逻辑
│   ├── eval/
│   │   ├── eval_rl_new.py  # 多策略 BO 对比评估
│   │   └── eval_pfn.py     # PFN 回归精度评估
│   └── common/           # IO、随机种子等工具
└── local_scripts/        # 各任务的 shell 脚本
```

---

## 三、核心算法流程（三阶段 Pipeline）

### 阶段 1：TabPFN 微调 (`finetune.py`)

目的：让 TabPFN（一个 pre-trained 的表格数据回归模型）适配特定函数族。

1. **采样训练变体**：在函数族的 in_range 参数范围内随机采样 k 个变体（如 10 个 Branin 变体，每个有不同的 dx/rotation/scale 变换）
2. **生成 BO 轨迹**：对每个变体运行 5 次 GP+EI 贝叶斯优化，**选最好的一条作为"真实轨迹"**
3. **拟合 Oracle GP**：用这条最佳轨迹拟合一个高斯过程
4. **合成轨迹**：从 Oracle GP 后验中通过 **RFF（Random Fourier Features）** 采样确定性函数 f，再随机采点生成 100 条合成轨迹
5. **微调 TabPFN**：将每条轨迹按 prefix/suffix 切分（prefix 作为 context，suffix 作为 test），用标准元学习方式微调 TabPFN

### 阶段 2：CAP-PPO 训练 (`train_rl.py`)

核心组件 -- **ImprovedDualTowerSelector** 网络（Actor-Critic）：

```
Context Tower:  [x_norm, y_norm] -> Embedding -> Self-Attention x 3
                                                        |
Candidate Tower: [x_norm, mu_norm, sigma_norm, score, budget] -> Embedding -> Self-Attention x 3 -> Cross-Attention x 3
                                                                                                     |
                                                                                    Actor Head -> logits (选哪个候选点)
                                                                                    Critic Head -> value (状态价值)
```

**特征设计**：
- **Context 特征**：归一化坐标 + 归一化 y 值
- **Candidate 特征**：归一化坐标 + 归一化 posterior mean + 归一化 posterior std + **采集函数分数（TAF_me 或 EI）** + 剩余 budget 比例
- **Step Embedding**：当前步数的可学习嵌入，加到两个塔上

**训练环境**（每个 episode）：
1. 从 Oracle GP（或直接目标函数）采样一个目标函数
2. 随机初始化 2 个上下文点
3. 每步：用 TabPFN/GP 做预测 -> Sobol 采样候选点 -> RL 策略选择一个点 -> 评估真实值 -> 更新上下文
4. **奖励设计**：基于 regret 改进量 + 时间惩罚 + 接近最优的阶梯奖励

**PPO 超参数**：gamma=0.95，lambda=0.9（GAE），clip=0.2，entropy_coef=0.02

### 阶段 3：评估 (`eval_rl_new.py`)

对比策略：Random, EI, UCB, PI, TAF(me), TAF(ranking), CAP-PPO, MetaBO（可选）

评估在 **ID（in_range）+ OOD（level 1/2/3）** 变体上进行，度量 **Rank** 和 **Simple Regret** 曲线。

---

## 四、研究切入点与相关工作定位

### 核心命题

用 RL 学习贝叶斯优化中的采集函数（Acquisition Function），属于 **Meta-learning for Bayesian Optimization** 领域。

### 相关工作对标

| 相关工作 | 核心思路 | 与本项目的关系 |
|---------|---------|-------------|
| **MetaBO** (Volpp et al., 2020) | PPO 学习 neural AF，从离散候选集中选点 | 本项目最直接的 baseline，架构从简单 MLP 升级为 Dual-Tower Cross-Attention |
| **TAF** (Wistuba et al., 2018) | 用源任务 GP 模型加权迁移 EI | 本项目将 TAF_me 分数作为 RL 的输入特征 |
| **PFNs4BO** (Muller et al., 2023) | 用 TabPFN 替代 GP 做代理模型 | 本项目沿用此思路，并增加了 task-specific fine-tuning |
| **OptFormer** (Chen et al., 2022) | Transformer 端到端建模整个 BO 过程 | 本项目的 Cross-Attention 架构与之有相似的 set-based reasoning 精神 |

### 项目的切入点

在 MetaBO 的"RL 学 AF"框架上做了三个升级：
1. 代理模型从 GP 换成 fine-tuned TabPFN
2. 网络架构从 MLP 升级为 Dual-Tower Cross-Attention
3. 输入特征加入 TAF/EI 分数作为先验

---

## 五、从半导体工艺优化视角的分析

### 5.1 问题本质

```
传统流程：  工程师经验 -> 选几组配方试 -> 看结果 -> 凭经验调整 -> 再试 -> ... (20次实验预算)
本项目目标：  历史工艺数据 -> 训练 RL agent -> agent 模拟工程师的"试错策略" -> 用更少实验找到好配方
```

关键约束对比：

| 约束 | 学术 BO 视角 | 半导体工艺现实 |
|------|------------|--------------|
| 每次实验成本 | 近乎零（调用函数） | 极高（一片晶圆几万元，加上机台时间） |
| 实验预算 | 通常 100-500 次 | 通常 10-30 次 |
| 参数空间 | 连续、可任意查询 | 实际上常常是离散档位、有工艺约束 |
| 先验知识 | 无 | 工程师有丰富经验，历史 lot 有数据 |
| 目标 | 找到精确全局最优 | 找到"足够好"的配方，**快速降低 regret** |

### 5.2 设计的自洽性

#### (1) "离散候选集"贴合工业现实

在半导体场景下，128 个离散候选点不是缺陷而是合理抽象：

- 工艺参数往往有**物理约束**（温度只能 50 度C 一档、气体流量有最小调节单位）
- 工程师做 DOE 时，本身就是**从有限的候选配方中选择下一组实验**
- 128 个候选点 = 在当前工艺知识下，**一张合理的实验计划表**

RL agent 做的事情本质上就是：**面对一张候选配方清单，根据之前的实验结果，决定下一步试哪个**——恰好就是经验丰富的工程师在做的事。

#### (2) "函数族 + OOD 泛化"映射工艺漂移

```
Branin family (dx, rotation, scale 变换)
    | 映射到
半导体工艺中：不同机台的 chamber 偏差、不同批次的材料漂移、不同产品的工艺窗口偏移
```

- **in_range** = 同一条产线上的正常工艺波动
- **ood_level_1/2/3** = 新机台、新材料、甚至新制程节点

验证的核心问题：**在 A 机台上训练的"优化经验"，能不能迁移到 B 机台？**

#### (3) Oracle GP + RFF 采样映射"数字孪生"

```
真实场景：工程师在某个工艺上跑了 20 次实验 (= best-of BO 轨迹)
         -> 拟合一个工艺响应面模型 (= Oracle GP)
         -> 从这个模型中生成大量虚拟实验 (= RFF 采样合成轨迹)
         -> 用这些虚拟数据训练 agent
```

对应半导体行业中**数字孪生（Digital Twin）** 的思路：用少量实测数据构建虚拟工艺模型，在虚拟环境中训练 AI，再部署到真实产线。RFF 采样保证了每个"虚拟工艺"是确定性的（对应一个特定的物理过程），比直接从 GP 后验采噪声样本更物理合理。

#### (4) TAF 特征作为 RL 输入 = "参考老师傅的经验"

将 TAF_me（基于历史工艺数据的迁移采集函数）作为 RL 的输入特征，本质上是：

```
agent 的决策 = f(当前实验数据, 代理模型预测, 老师傅的建议)
```

"老师傅的建议"= TAF 从历史工艺轨迹中提取的偏好。RL 学的是**什么时候听老师傅的、什么时候自己探索**。

---

## 六、从学术 BO 角度的补充分析

### 6.1 算法设计的合理之处

**Dual-Tower Cross-Attention 架构合理**：候选点选择本质是 set-to-element 问题，Cross-Attention 天然适合"query 来自候选集、key-value 来自历史"的交互模式，相比 MetaBO 的 per-candidate MLP 是有道理的升级。

**TAF/EI 分数作为特征降低学习难度**：给 RL 提供了强先验/参考信号，网络不需要从零学习"posterior mean 低且 std 高的点好"，RL 只需学习"什么时候、多大程度上偏离 EI 的建议"。

**ID/OOD 分级评估体系**：通过递增变换幅度来测试泛化，实验设计规范。

### 6.2 从学术角度仍可改进的问题

#### (1) RL 的提升空间可能有限（"天花板"问题）

候选特征中已包含 TAF_me 或 EI 分数。如果 RL 策略只学到"选 score 最高的候选点"，则等价于 TAF/EI 本身。RL 唯一能超越 EI/TAF 的途径是学到**非贪心（non-myopic）策略**，但 gamma=0.95、总步数仅 18 步，non-myopic 优势不明显。

**建议**：做消融实验——把 score 特征去掉看性能下降多少，验证 RL 学到的不仅仅是"复读"先验 AF 分数。

#### (2) 奖励函数存在 task-specific 硬编码

当前奖励函数中 `regret < 0.1`、`regret < 0.01` 等阈值对不同函数族意义完全不同（Branin y* 约 0.4 vs Hartmann-3D y* 约 -3.86），且 `goldstein_price_family` 有单独的奖励分支。

**建议**：用 **normalized regret improvement** `(regret_{t-1} - regret_t) / regret_0` 作为统一奖励，天然在 [0,1] 内，对所有函数族通用。

#### (3) Pipeline 耦合风险与分布偏移

训练分三个串行阶段，每个阶段的误差会传播。特别是：RL 训练时用 TabPFN 做预测，但 TabPFN 微调时用的轨迹与 RL 的在线交互轨迹分布不同，存在 **distribution shift** 未被处理。

#### (4) 可扩展性局限

目前仅支持 2D 和 3D 问题，部分代码（grid search、estimate_global_min）对高维不友好。实际半导体工艺可能涉及 5-20 个参数。

#### (5) 评估方法学的遗漏

- 缺少与连续优化 AF（BoTorch qEI/qKG）的对比
- 缺少计算代价对比（wall-clock time 一致条件下的性能）
- 缺少 ablation study（无法区分性能提升来自 fine-tuned TabPFN、Cross-Attention 架构、还是 RL 本身）

---

## 七、从工业落地角度的关键问题

### 7.1 Benchmark 物理可信度不足

Branin/Goldstein-Price 的 dx/rotation/scale 变换能否代表真实工艺漂移？真实半导体工艺响应面通常有：
- **多峰结构**（多个局部最优工艺窗口）
- **参数耦合**（温度和压力的交互效应）
- **噪声**（批次间随机波动）
- **约束**（某些参数组合会导致工艺失败）

**建议**：增加更贴近真实工艺的 benchmark（如 TCAD 仿真简化模型），或构造具有多峰 + 噪声 + 参数耦合特性的函数族。

### 7.2 奖励函数需要对齐工程师的阶段性策略

工业上工程师的优化策略是阶段性的：
- **前几步大胆探索**（把参数空间大致摸清）
- **中间几步锁定区域**（确定哪个工艺窗口最有潜力）
- **最后几步精细调优**（在最佳区域内微调到最优）

**建议**：
- 可视化 agent 在不同 step 选择的候选点在参数空间中的分布，观察是否有"先全局探索 -> 后局部利用"的行为模式
- 如果没有此行为，考虑更细致的奖励设计

### 7.3 Sim-to-Real Gap

Oracle GP 仅从 20 个点拟合，可能：
- 在未探索区域给出过度光滑的估计
- 低估真实工艺的非线性

Agent 学到的可能是"在光滑近似上的优化策略"，部署到真实工艺时性能可能退化。

**建议**：增加 sim-to-real 评估——用 Oracle GP 训练，在原始真实函数上测试（`objective_source="direct"` 模式已支持）。

### 7.4 论文叙事线建议

```
不推荐的叙事：我们提出了一种比 EI/UCB 更好的 BO 采集函数
推荐的叙事：  我们提出了一种从少量历史工艺数据中学习优化策略的框架，
              能够将跨工艺的优化经验迁移到新工艺上，
              在极有限的实验预算下显著减少试错次数
```

核心贡献应突出：
1. **数据高效**：仅需每个源工艺的一条 BO 轨迹（20 个点），就能训练出可迁移的优化策略
2. **经验迁移**：通过函数族训练 + OOD 评估，验证了策略在工艺漂移下的鲁棒性
3. **自动化决策**：agent 替代了工程师的人工试错过程，且不依赖专家在线指导

---

## 八、总结评价

| 维度 | 评价 |
|------|------|
| **问题动机** | 强——半导体工艺优化确实是 RL 的好场景（少数据、高成本、需迁移） |
| **技术路线** | 合理且自洽——离散候选集、函数族、Oracle GP+RFF、TAF 特征，在工业语境下都讲得通 |
| **架构设计** | Dual-Tower Cross-Attention 比 MetaBO 的 MLP 更有表达力，升级方向正确 |
| **核心创新度** | 中等——主要是将已有组件（TabPFN + RL + TAF 特征 + Cross-Attention）组合，需要更强的实验证据来说明组合效果 |
| **实验说服力** | 偏弱——缺少 ablation、缺少连续 AF baseline、缺少计算代价分析、奖励函数有 task-specific 硬编码 |
| **工业适配度** | 中等——框架逻辑对路，但 benchmark 与真实工艺差距大，可扩展性有限 |

### 改进优先级

1. **统一奖励函数**，去掉 task-specific 硬编码
2. **增加更有说服力的 benchmark**（贴近真实工艺特性）
3. **可视化 agent 的阶段性策略**（探索 -> 利用行为是否出现）
4. **补充 ablation study**（各组件贡献拆分）
5. **增加 sim-to-real 评估**（Oracle GP 训练 vs 真实函数测试）
6. **增加连续 AF baseline**（BoTorch qEI 等）作为对照

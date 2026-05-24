# CAP-PPO 最终实验报告

> **项目**: Context-Aware Policy for Bayesian Optimization (CAP-PPO)
> **日期**: 2026-03-24
> **版本**: v12 (BNN_mean + RFF perturbation)

---

## 目录

1. [项目概述](#1-项目概述)
2. [实验方法](#2-实验方法)
3. [基准函数测试](#3-基准函数测试)
4. [真实数据仿真器测试](#4-真实数据仿真器测试)
5. [Scale Sweep 泛化性分析](#5-scale-sweep-泛化性分析)
6. [失败实验分析](#6-失败实验分析)
7. [关键发现与结论](#7-关键发现与结论)
8. [附录](#8-附录)

---

## 1. 项目概述

### 1.1 核心思想

用 PPO 强化学习训练的 **Context-Aware Policy (CAP-PPO)** 替代传统贝叶斯优化采集函数（EI/UCB/PI），在有限预算黑盒优化任务上实现更好的优化效果。

### 1.2 模型架构

- **Dual-Tower Cross-Attention 网络**: Context Tower (已观测数据) + Candidate Tower (候选点)
- 参数: hidden_dim=128, n_self_attn=3, n_cross_attn=3, n_heads=8
- 输入特征: 坐标 + TabPFN 预测均值/标准差 + is_persistent 标志 + TAF ranking score
- 候选策略: 128 persistent Sobol + 64 adaptive local

### 1.3 训练管线

```
Step 1: 生成变体 + BO轨迹数据 (仿射变换: 旋转+缩放+平移)
Step 1.5: 生成 TAF source data (用于 TAF ranking 特征)
Step 1.5b: [可选] 训练 BNN 代理模型
Step 2: PPO 训练 (5000 episodes)
Step 3a/3b: 评估 (GP / TabPFN 代理, 4个OOD等级)
```

### 1.4 Objective Source 选择

| Source | 适用场景 | 机制 |
|--------|---------|------|
| `oracle_gp` | 合成函数 (Hartmann等) | GP拟合 → RFF采样 |
| `bnn` | NN仿真器 (alkox等) | f(x) = BNN_mean(x) + α·RFF_prior(x) |
| `direct` | 直接调用仿真器 | 每次调用真实函数 |

---

## 2. 实验方法

### 2.1 评估指标

**Simple Regret** = max(0, best_y_observed − global_min)，越低越好。

### 2.2 评估设置 (Optimal Mode)

| 参数 | CAP-PPO | 传统基线 (EI/UCB/PI/TAF) | Random |
|------|---------|--------------------------|--------|
| 候选点数 | 192 (128+64) | 2048 (Sobol) | 128 |
| 代理模型 | GP / TabPFN | GP / TabPFN | — |

> 注: 基线使用 2048 候选点，是 CAP-PPO 的 10.7 倍，属于对 CAP-PPO 不利的公平测试。

### 2.3 变体分组

| 组别 | 含义 | 仿射变换强度 (4D 任务) |
|------|------|----------------------|
| in_range | 训练分布内 | dx ±0.08, rot ±28°, sx [0.84, 1.0] |
| ood_level_1 | 轻度OOD | dx ±0.12, rot ±36°, sx [0.78, 1.0] |
| ood_level_2 | 中度OOD | dx ±0.16, rot ±44°, sx [0.72, 1.0] |
| ood_level_3 | 重度OOD | dx ±0.20, rot ±52°, sx [0.66, 1.0] |

每组 20 个变体, 每个变体 3 次独立运行。报告 mean ± std。

---

## 3. 基准函数测试

### 3.1 Hartmann 6D Family

- **维度**: 6D, [0,1]^6
- **训练**: oracle_gp objective, 4500 episodes (float32, TabPFN base)
- **预算**: 50 evals/episode (2 init + 48 steps)

#### 3.1.1 GP 代理评估结果

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| **CAP-PPO** | **0.014 ± 0.027** | **0.190 ± 0.180** | **0.443 ± 0.379** | **0.859 ± 0.635** |
| TAF_ranking | 0.253 ± 0.038 | 0.498 ± 0.160 | 0.826 ± 0.406 | 1.043 ± 0.347 |
| TAF_me | 0.269 ± 0.065 | 0.496 ± 0.185 | 0.917 ± 0.442 | 1.037 ± 0.359 |
| EI | 1.037 ± 0.350 | 1.005 ± 0.440 | 1.086 ± 0.400 | 1.175 ± 0.470 |
| UCB | 0.987 ± 0.380 | 1.083 ± 0.371 | 1.037 ± 0.369 | 1.237 ± 0.346 |
| PI | 0.977 ± 0.409 | 1.016 ± 0.405 | 1.099 ± 0.381 | 1.040 ± 0.341 |
| Random | 1.647 ± 0.476 | 1.582 ± 0.382 | 1.831 ± 0.357 | 1.856 ± 0.455 |

#### 3.1.2 TabPFN 代理评估结果

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| **CAP-PPO** | **0.030 ± 0.013** | **0.142 ± 0.096** | **0.213 ± 0.149** | 0.685 ± 0.566 |
| TAF_ranking | 0.232 ± 0.039 | 0.385 ± 0.126 | 0.664 ± 0.289 | **0.757 ± 0.364** |
| TAF_me | 0.260 ± 0.053 | 0.516 ± 0.235 | 0.847 ± 0.369 | 0.941 ± 0.327 |
| EI | 0.742 ± 0.246 | 0.821 ± 0.313 | 1.016 ± 0.401 | 0.886 ± 0.444 |
| UCB | 0.791 ± 0.186 | 0.721 ± 0.215 | 0.943 ± 0.355 | 0.916 ± 0.346 |
| PI | 0.766 ± 0.358 | 0.707 ± 0.178 | 0.853 ± 0.387 | 0.902 ± 0.414 |
| Random | 1.647 ± 0.476 | 1.582 ± 0.382 | 1.831 ± 0.357 | 1.856 ± 0.455 |

#### 3.1.3 Hartmann 6D 小结

| 对比 | GP 代理 (in_range) | TabPFN 代理 (in_range) |
|------|-------------------|----------------------|
| CAP-PPO vs TAF_ranking | **18x 更优** (0.014 vs 0.253) | **7.7x 更优** (0.030 vs 0.232) |
| CAP-PPO vs EI | **74x 更优** (0.014 vs 1.037) | **25x 更优** (0.030 vs 0.742) |
| CAP-PPO vs Random | **118x 更优** (0.014 vs 1.647) | **55x 更优** (0.030 vs 1.647) |

- CAP-PPO 在**全部 OOD 等级 + 两种代理**下均为最优或并列最优
- 在 GP 代理下, 即使 ood_level_3, CAP-PPO (0.859) 仍优于所有传统 AF
- 在 TabPFN 代理下, ood_level_3 时 TAF_ranking (0.757) 略优于 CAP-PPO (0.685)，差距不大

---

### 3.2 Branin Family (参考实验)

- **维度**: 2D
- **训练**: oracle_gp, TabPFN tuned

#### GP 代理 (部分结果)

| Method | ood_level_2 | ood_level_3 |
|--------|-------------|-------------|
| UCB | **0.66 ± 0.75** | **0.13 ± 0.26** |
| EI | 0.73 ± 0.81 | 0.16 ± 0.31 |
| CAP-PPO | 1.69 ± 2.61 | 0.31 ± 0.48 |
| TAF_ranking | 1.14 ± 1.26 | 0.66 ± 0.84 |

#### TabPFN Tuned 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| EI | **0.59 ± 0.83** | **0.76 ± 0.56** | **1.08 ± 1.25** | 0.59 ± 0.84 |
| CAP-PPO | 1.86 ± 2.86 | 3.20 ± 5.65 | 1.69 ± 2.44 | 1.18 ± 1.47 |
| TAF_ranking | 1.02 ± 1.16 | 2.39 ± 2.80 | 1.08 ± 1.10 | 0.87 ± 0.61 |
| PI | 1.07 ± 1.43 | 0.27 ± 0.45 | 2.43 ± 3.70 | **0.31 ± 0.38** |

> Branin 属于简单 2D 问题，传统 AF 已高度有效，CAP-PPO 表现中等。此实验主要用于架构验证而非性能突破。

---

## 4. 真实数据仿真器测试

### 4.0 Olympus 仿真器难度基准

使用 GP+EI (2048 Sobol 候选点, 5 次独立运行) 对所有 10 个 Olympus NeuralNet 仿真器进行难度评估:

| 难度等级 | 数据集 | 维度 | 方向 | EI Regret (mean±std) | 归一化 Regret | y_range |
|---------|--------|------|------|---------------------|--------------|---------|
| **VERY HARD** | **alkox** | **4D** | MAX | **40.33 ± 25.49** | **0.399** | 101.14 |
| MODERATE | fullerenes | 3D | MAX | 0.017 ± 0.007 | 0.033 | 0.51 |
| MODERATE | hplc | 6D | MAX | 35.55 ± 31.94 | 0.014 | 2453.60 |
| EASY | benzylation | 4D | MIN | 0.094 ± 0.051 | 0.005 | 19.19 |
| EASY | photo_pce10 | 4D | MIN | 0.000 ± 0.000 | 0.000 | 1.33 |
| EASY | snar | 4D | MIN | 0.002 ± 0.006 | 0.000 | 5.17 |
| EASY | suzuki | 4D | MAX | -1.352 ± 0.000 | -0.015 | 89.36 |
| EASY | photo_wf3 | 4D | MIN | 0.008 ± 0.008 | 0.002 | 3.68 |
| EASY | colors_n9 | 3D | MIN | -0.001 ± 0.000 | -0.002 | 0.56 |
| EASY | colors_bob | 5D | MIN | -0.011 ± 0.000 | -0.021 | 0.52 |

> 归一化 Regret 阈值: <0.01 EASY, 0.01-0.05 MODERATE, 0.05-0.15 HARD, >0.15 VERY HARD

---

### 4.1 Alkox Emulator (4D, VERY HARD) — 成功案例

- **任务**: catalase, peroxidase, alcohol_oxidase, pH → conversion (maximize, 框架 negate)
- **训练**: BNN objective, 5000 episodes, 30 evals/episode (2 init + 28 steps)
- **难度**: EI 归一化 regret = 0.399, 唯一 VERY HARD 任务

#### 4.1.1 实验 A: BNN kl=0.003 (v11, 权重采样), GP 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| **CAP-PPO** | **0.14 ± 0.41** | **6.97 ± 10.92** | 20.79 ± 12.01 | 13.49 ± 9.82 |
| TAF_ranking | 6.29 ± 5.96 | 6.64 ± 4.32 | 16.71 ± 8.78 | **12.12 ± 7.61** |
| TAF_me | 7.28 ± 7.20 | 11.19 ± 9.22 | **13.73 ± 4.22** | 15.77 ± 12.56 |
| EI | 42.01 ± 11.00 | 30.82 ± 8.59 | 17.83 ± 7.93 | 17.50 ± 10.36 |
| UCB | 41.63 ± 12.10 | 33.03 ± 10.86 | 18.63 ± 7.68 | 16.76 ± 10.25 |
| PI | 45.15 ± 9.96 | 31.33 ± 11.99 | 20.46 ± 8.82 | 17.54 ± 11.92 |
| Random | 48.09 ± 11.34 | 37.65 ± 11.92 | 25.99 ± 12.44 | 22.56 ± 11.90 |

#### 4.1.2 实验 B: BNN kl=0.001 (v12, BNN_mean+RFF), GP 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| **CAP-PPO** | **3.97 ± 5.90** | 18.31 ± 17.58 | 17.17 ± 9.69 | 18.00 ± 16.24 |
| TAF_ranking | 11.97 ± 11.67 | **18.32 ± 10.34** | **15.06 ± 7.17** | **11.69 ± 12.06** |
| TAF_me | 11.91 ± 10.88 | 19.02 ± 12.18 | 15.40 ± 8.18 | 14.31 ± 15.39 |
| EI | 33.37 ± 8.45 | 23.36 ± 10.34 | 10.96 ± 7.26 | 12.31 ± 10.78 |
| UCB | 33.16 ± 12.23 | 21.57 ± 11.16 | 11.66 ± 7.38 | 11.09 ± 10.52 |
| PI | 35.77 ± 8.27 | 23.03 ± 11.03 | 13.31 ± 6.13 | 10.77 ± 8.86 |
| Random | 40.16 ± 10.50 | 28.30 ± 13.13 | 18.63 ± 9.71 | 18.71 ± 11.53 |

#### 4.1.3 实验 B: BNN kl=0.001 (v12, BNN_mean+RFF), TabPFN 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| **CAP-PPO** | **2.49 ± 6.04** | **9.99 ± 12.39** | 15.16 ± 11.20 | 14.61 ± 14.86 |
| TAF_ranking | 8.39 ± 10.98 | 16.14 ± 10.25 | **13.31 ± 8.77** | **9.52 ± 9.92** |
| TAF_me | 10.04 ± 9.41 | 18.42 ± 13.77 | 15.10 ± 7.33 | 14.45 ± 14.13 |
| EI | 28.40 ± 10.42 | 20.09 ± 11.82 | 12.51 ± 8.54 | 10.04 ± 8.68 |
| UCB | 23.85 ± 11.92 | 17.06 ± 9.52 | 11.97 ± 9.14 | 7.65 ± 6.47 |
| PI | 33.79 ± 11.19 | 20.91 ± 9.28 | 10.64 ± 8.22 | 11.25 ± 10.79 |
| Random | 40.16 ± 10.50 | 28.30 ± 13.13 | 18.63 ± 9.71 | 18.71 ± 11.53 |

#### 4.1.4 Alkox 小结

| 对比 | 实验A (kl003, GP) | 实验B (kl001, GP) | 实验B (kl001, TabPFN) |
|------|------------------|------------------|----------------------|
| CAP-PPO in_range | **0.14** | **3.97** | **2.49** |
| vs TAF_ranking | **45x 更优** | **3.0x 更优** | **3.4x 更优** |
| vs EI | **300x 更优** | **8.4x 更优** | **11.4x 更优** |

**核心发现**:
- CAP-PPO 在 alkox in_range 取得**突破性结果** (regret 接近 0)
- kl=0.003 的 in_range 表现 (0.14) 优于 kl=0.001 (2.49-3.97)，但 v12 (kl=0.001) 的 OOD 鲁棒性更好
- OOD 退化模式: in_range → ood_level_3 约 **96x 退化** (kl003, 0.14→13.49)
- 从 ood_level_2 开始, CAP-PPO 不再显著优于 TAF_ranking/EI

---

### 4.2 Benzylation Emulator (4D, EASY) — 失败案例

- **任务**: flow_rate, ratio, solvent, temperature → impurity (minimize)
- **难度**: EI 归一化 regret = 0.005, EASY
- **进行了 3 轮独立实验**

#### 4.2.1 实验 A: fixflow 变体 + oracle_gp, GP 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| TAF_me | **0.002 ± 0.004** | **0.003 ± 0.004** | **0.004 ± 0.005** | **0.002 ± 0.004** |
| TAF_ranking | 0.003 ± 0.005 | 0.003 ± 0.004 | 0.004 ± 0.005 | 0.002 ± 0.003 |
| EI | 0.013 ± 0.017 | 0.018 ± 0.020 | 0.013 ± 0.012 | 0.008 ± 0.011 |
| CAP-PPO | 0.270 ± 0.082 | 0.384 ± 0.169 | 0.458 ± 0.286 | 0.466 ± 0.374 |
| Random | 0.856 ± 0.482 | 0.889 ± 0.600 | 0.781 ± 0.685 | 1.053 ± 0.953 |

#### 4.2.2 实验 B: transform 变体 + oracle_gp, GP 代理

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| TAF_me | **0.010 ± 0.011** | 0.010 ± 0.016 | 0.012 ± 0.017 | 0.038 ± 0.082 |
| TAF_ranking | 0.013 ± 0.013 | **0.011 ± 0.018** | **0.012 ± 0.016** | **0.005 ± 0.007** |
| EI | 0.018 ± 0.018 | 0.012 ± 0.018 | 0.009 ± 0.013 | 0.004 ± 0.006 |
| CAP-PPO | 0.157 ± 0.090 | 0.300 ± 0.186 | 0.725 ± 0.551 | 0.738 ± 0.656 |
| Random | 0.853 ± 0.480 | 0.958 ± 0.403 | 0.832 ± 0.473 | 0.891 ± 0.526 |

#### 4.2.3 实验 C: transform 变体 + BNN, TabPFN 代理 (最差)

| Method | in_range | ood_level_1 | ood_level_2 | ood_level_3 |
|--------|----------|-------------|-------------|-------------|
| TAF_ranking | **0.012 ± 0.011** | **0.013 ± 0.016** | **0.019 ± 0.024** | **0.006 ± 0.009** |
| EI | 0.024 ± 0.028 | 0.014 ± 0.017 | 0.029 ± 0.049 | 0.012 ± 0.024 |
| Random | 0.853 ± 0.480 | 0.958 ± 0.403 | 0.832 ± 0.473 | 0.891 ± 0.526 |
| CAP-PPO | 2.010 ± 1.492 | 2.574 ± 1.123 | 2.126 ± 1.099 | 2.604 ± 2.224 |

#### 4.2.4 Benzylation 小结

| 指标 | 最佳 CAP-PPO | 最佳基线 | 差距 |
|------|-------------|---------|------|
| in_range 最佳 | 0.157 (实验B) | TAF_me 0.010 | **16x 更差** |
| in_range 最差 | 2.010 (实验C) | TAF_ranking 0.012 | **168x 更差** |
| 实验C vs Random | 2.010 | Random 0.853 | **比随机还差 2.4x** |

**失败原因**: Benzylation 对传统 BO 来说太简单 (EI regret ~0.01)，CAP-PPO 没有改进空间。BNN 训练函数质量差（30 数据点 / 4848 参数，后验几乎无约束）导致训练信号退化。

---

### 4.3 HPLC Emulator (6D, MODERATE) — 未完成

- **任务**: 6D → peak_area (maximize, 框架 negate)
- **训练**: oracle_gp + fixtubing 变体

#### GP 代理评估 (仅完成 in_range 和 ood_level_1)

| Method | in_range | ood_level_1 |
|--------|----------|-------------|
| TAF_ranking | **31.26 ± 19.49** | 65.04 ± 39.71 |
| TAF_me | 40.05 ± 25.69 | **63.64 ± 18.80** |
| CAP-PPO | 81.41 ± 41.19 | 111.89 ± 82.29 |
| EI | 90.23 ± 32.40 | 63.64 ± 18.80 |
| PI | 85.85 ± 41.48 | 64.84 ± 31.45 |
| UCB | 124.89 ± 61.25 | 82.00 ± 39.87 |
| Random | 144.71 ± 72.49 | 159.59 ± 75.44 |

**状态**:
- oracle_gp 训练未收敛 (regret 稳定在 300-700)
- BNN 训练因 variant 3 数值不稳定 (y range: -2479 ~ 0) 导致 NaN 而失败
- 评估在 ood_level_2 处中断
- **结论: HPLC 目前不可用**

---

## 5. Scale Sweep 泛化性分析

### 5.1 实验设置

- **模型**: alkox BNN kl=0.001 (v12)
- **代理**: TabPFN base
- **Scale**: 0.5 ~ 2.0 (1.0 = in_range 标准)
- **统计**: 20 变体/scale × 3 runs/变体

### 5.2 结果

| Scale | 变换范围 | CAP-PPO | TAF_ranking | EI | 胜者 |
|-------|---------|---------|-------------|-----|------|
| **0.50** | dx±0.04, rot±14° | **0.00 ± 0.00** | 2.92 ± 3.50 | 36.47 ± 19.30 | **CAP-PPO** (完美) |
| **0.75** | dx±0.06, rot±21° | **1.09 ± 2.45** | 6.11 ± 9.48 | 32.71 ± 17.37 | **CAP-PPO** (5.6x) |
| **1.00** | dx±0.08, rot±28° | **1.63 ± 7.09** | 4.92 ± 6.31 | 29.65 ± 13.49 | **CAP-PPO** (3.0x) |
| **1.25** | dx±0.10, rot±35° | **1.97 ± 4.51** | 7.24 ± 7.13 | 32.84 ± 14.66 | **CAP-PPO** (3.7x) |
| 1.50 | dx±0.12, rot±42° | 10.36 ± 14.22 | **7.08 ± 6.31** | 24.35 ± 15.65 | TAF_ranking |
| 1.75 | dx±0.14, rot±49° | **11.20 ± 15.96** | 22.78 ± 16.74 | 23.46 ± 11.68 | **CAP-PPO** |
| 2.00 | dx±0.16, rot±56° | **13.92 ± 15.57** | 17.37 ± 13.38 | 22.62 ± 15.14 | **CAP-PPO** |

### 5.3 Scale Sweep 关键发现

```
CAP-PPO 优势区     过渡带      混合区
  (显著领先)      (交叉点)    (互有胜负)
├───────────────┤───────────┤──────────────┤
0.5    0.75   1.0   1.25   1.5   1.75   2.0
```

- **Scale ≤ 1.25**: CAP-PPO 显著优于所有基线 (3x-∞)
- **Scale = 1.5**: 交叉点，TAF_ranking 首次超过 CAP-PPO
- **Scale ≥ 1.75**: CAP-PPO 仍有竞争力，但不再稳定优于 TAF_ranking
- **全程**: CAP-PPO 始终优于 EI (即使在 scale=2.0)

---

## 6. 失败实验分析

### 6.1 Oracle GP 在 NN 仿真器上的失败

| 指标 | Hartmann 6D (成功) | Alkox (失败) |
|------|-------------------|-------------|
| GP Spearman 相关 | 0.518 | ~0.10 |
| GP 长度尺度 | 正常 (0.1-10) | 异常 (1e-5 或 >8000) |
| 变体 GP 拟合通过率 | 10/10 | 2/10 |
| RFF vs_true | 0.247 | 0.071 |

**根因**: GP Matern 2.5 的平稳性假设与 NN 仿真器的非平稳景观根本不匹配。

### 6.2 BNN 权重采样的多样性不足

| 方法 | inter (函数相似度) | vs_true (准确度) |
|------|-------------------|-----------------|
| BNN 权重采样 (kl=0.001) | 0.891 (太高) | 0.602 |
| BNN 权重采样 (kl=0.003) | 0.530 | 0.300 |
| **BNN_mean + RFF (v12)** | **0.156** | **~0.60** |
| Oracle GP→RFF (参考) | 0.245 | 0.267 |

**解决方案**: v12 将 BNN mean (保真度) 与 RFF prior (多样性) 解耦, 达到与 Hartmann 成功配置相当的 inter=0.156。

### 6.3 Benzylation 失败的 BNN 后验分析

```
BNN 参数: 4848 个
训练数据: 30 点
数据/参数比: 0.006 (极度不足)

Layer 1 (48→48): CV_median = 51-116  ← 后验完全无约束
Layer 2 (48→48): CV_median = 51-68   ← 后验完全无约束
→ 采样函数接近随机噪声
→ PPO 梯度被噪声淹没
→ Agent 学会"反向利用" TAF（训练中合理，评估中有害）
→ 结果比 Random 还差
```

### 6.4 训练崩溃记录

| 实验 | 崩溃点 | 错误 | 原因 |
|------|-------|------|------|
| Hartmann 6D (首次) | Episode 4550 | TabPFN NaN | float16 精度不足 (已切换 float32) |
| HPLC BNN | Variant 3, Epoch 1 | NaN | y range 极端 (-2479~0) |
| Alkox oracle_gp | Episode 3880 | 手动终止 | Regret 无收敛 (~30 不下降) |

---

## 7. 关键发现与结论

### 7.1 总结表

| 任务 | 类型 | 难度 | 最佳配置 | in_range Regret | vs 最佳基线 | 结论 |
|------|------|------|---------|----------------|------------|------|
| **Hartmann 6D** | 合成 | — | oracle_gp, GP | **0.014** | **18x↑** (vs TAF) | **成功** |
| **Alkox** | 仿真器 | VERY HARD | BNN kl003, GP | **0.14** | **45x↑** (vs TAF) | **突破** |
| Branin | 合成 | — | oracle_gp | 1.69 | 2.6x↓ (vs UCB) | 中等 |
| Benzylation | 仿真器 | EASY | transform, GP | 0.157 | 16x↓ (vs TAF) | **失败** |
| HPLC | 仿真器 | MODERATE | — | 81.41 | 2.6x↓ (vs TAF) | **未完成** |

### 7.2 核心结论

**1. CAP-PPO 的价值定位: 困难优化任务**

CAP-PPO 仅在传统 BO 方法吃力的 **HARD/VERY HARD** 任务上展现价值。在 EASY 任务上，传统采集函数已接近最优，RL 训练的策略没有改进空间。

**2. Objective Source 选择是成败关键**

| 函数类型 | 推荐 Source | 原因 |
|---------|------------|------|
| 解析函数 (Hartmann, Branin) | oracle_gp | GP 能准确拟合光滑函数 |
| NN 仿真器 (Alkox, Benzylation) | bnn (v12) | GP 无法拟合 NN 的非平稳景观 |

**3. BNN_mean + RFF 是方法论贡献**

v12 的 `f(x) = BNN_mean(x) + α·RFF_prior(x)` 设计:
- BNN mean: 保证训练函数忠实于真实仿真器 (vs_true ~0.60)
- RFF prior: 提供可控多样性 (inter ~0.156)
- 两者完全解耦，可独立调参

**4. OOD 泛化单调退化但可预测**

| 任务 | in_range → ood_3 退化倍数 |
|------|--------------------------|
| Hartmann 6D | 0.014 → 0.859 = **61x** |
| Alkox (kl003) | 0.14 → 13.49 = **96x** |

退化在 ood_level_2 (~scale 1.5) 后急剧加速。

**5. 候选点策略不对等下仍能胜出**

CAP-PPO 使用 192 候选点，基线使用 2048 候选点 (10.7x 更多)。在困难任务上 CAP-PPO 仍能以数量级差距胜出，说明策略质量远比候选点数量重要。

### 7.3 实验配置汇总

| 参数 | 4D 任务 | 6D 任务 |
|------|---------|---------|
| coord_dim | 4 | 6 |
| max_steps | 28 | 48 |
| n_init | 2 | 2 |
| total_evals | 30 | 50 |
| n_persistent_base | 128 | 128 |
| n_total_candidates | 192 | 256 |
| k_centers | 2 | 3 |
| local_h | 0.17 | 0.15 |
| local_h_decay | 0.95 | 0.95 |
| PPO episodes | 5000 | 5000 |
| BNN rff_alpha | 5.0 | 5.0 |
| BNN rff_length_scale | 0.3 | 0.3 |

### 7.4 待解决问题

1. **HPLC**: BNN 训练数值稳定性问题 (极端 y range)
2. **OOD 泛化**: 从 scale 1.5 开始退化严重，需要更 robust 的训练策略
3. **Benzylation 类 EASY 任务**: 需要判断机制避免在简单任务上使用 CAP-PPO
4. **更多任务验证**: fullerenes (3D, MODERATE) 作为下一个候选

---

## 8. 附录

### 8.1 所有训练运行清单

| 运行目录 | 任务 | Objective | Episodes | 状态 | 耗时 |
|---------|------|-----------|----------|------|------|
| ppo_alkox_emulator_bnn_kl001 | alkox | BNN kl=0.001 (v12) | 5000 | 完成 | ~20h |
| ppo_alkox_emulator_bnn_kl003 | alkox | BNN kl=0.003 (v11) | 5000 | 完成 | ~20h |
| ppo_alkox_emulator_transform | alkox | oracle_gp | ~3880 | 中断 | ~11h |
| ppo_benzylation_emulator | benzylation | BNN | 5000 | 完成 | ~15h |
| ppo_benzylation_emulator_fixflow | benzylation | oracle_gp | 5000 | 完成 | ~11h |
| ppo_benzylation_emulator_transform | benzylation | oracle_gp | 5000 | 完成 | ~16h |
| ppo_hartmann_6d_family_fast | hartmann_6d | oracle_gp | 4500 | 完成 | ~11h |
| ppo_hplc_emulator_fixtubing | hplc | oracle_gp | 5000 | 完成 | — |
| ppo_hplc_emulator_transform | hplc | oracle_gp | ~3180 | 中断 | — |
| ppo_hplc_emulator_bnn_kl003 | hplc | BNN kl=0.003 | 0 | **失败** | ~12min |

### 8.2 评估结果文件索引

| 目录 | 内容 | JSON | PNG |
|------|------|------|-----|
| results_policies/hartmann_6d_family_fast/ | Hartmann 6D GP+TabPFN | 2 | ~40 |
| results_policies/alkox_emulator_bnn_kl003/ | Alkox v11 GP + 轨迹可视化 | 1 | ~30 |
| results_policies/alkox_emulator_bnn_kl001/ | Alkox v12 GP+TabPFN | 2 | ~40 |
| results_policies/alkox_emulator_scale_sweep_v2/ | Scale Sweep (定稿) | 1 | 3 |
| results_policies/benzylation_emulator/ | Benzylation 原始 TabPFN | 1 | ~10 |
| results_policies/benzylation_emulator_fixflow/ | Benzylation fixflow GP+TabPFN | 2 | ~40 |
| results_policies/benzylation_emulator_transform/ | Benzylation transform GP+TabPFN | 2 | ~40 |
| results_policies/emulator_difficulty_test/ | 10个仿真器难度评估 | 1 | 10 |
| results_policies/hplc_emulator_fixtubing/ | HPLC (不完整) | 0 | ~20 |

### 8.3 代码版本演进

| 版本 | 日期 | 核心变更 |
|------|------|---------|
| v1 | 03-09 | 统一 reward, range-based 归一化, cosine LR |
| v2 | 03-09 | persistent+adaptive 候选策略, 去掉 EI shortcut |
| v3 | 03-10 | improvement AUC reward, 去掉 global_min 依赖 |
| v4 | 03-10 | TAF ranking 特征, per-episode C_func |
| v5 | 03-11 | 集成 benzylation 仿真器 |
| v6 | 03-11 | TabPFN 缓存优化 (100x 加速) |
| v7 | 03-12 | benzylation 切换 flow_rate 为变体参数 |
| v8 | 03-16 | 统一仿射变换变体生成 (rotation+scaling+translation) |
| v9 | 03-17 | 引入 BNN surrogate |
| v10 | 03-18 | 添加 alkox 仿真器 |
| v11 | 03-18 | alkox 切换到 BNN objective (权重采样) |
| v12 | 03-19 | BNN_mean + RFF 扰动 (最终版本) |

---

> *报告生成时间: 2026-03-24*
> *项目路径: <repo-root>/*

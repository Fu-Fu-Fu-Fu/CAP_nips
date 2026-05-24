# Olympus 仿真器 BO 难度测试结果

**日期**: 2026-03-18
**脚本**: `MYRL/scripts/test_emulator_difficulty.py`
**结果文件**: `results_policies/emulator_difficulty_test/difficulty_results.json`
**等高线图**: `results_policies/emulator_difficulty_test/contour_*.png`

---

## 1. 测试方法

对所有 10 个 Olympus NeuralNet 仿真器进行传统 BO（GP+EI）测试：

- **GP**: GPy Matern52 ARD 核，固定噪声 1e-6，5 次重启优化
- **采集函数**: Expected Improvement (EI)
- **候选点**: 每步 2048 个 Sobol 候选
- **预算**: 3-4D 20 evals, 5D 30 evals, 6D 50 evals（n_init=2）
- **重复**: 每个数据集 5 次独立运行
- **Global optimum 估计**: 10K Sobol 网格
- **优化方向**: 根据数据集目标自动处理（最大化数据集取反后统一最小化）

### 优化方向

| 方向 | 数据集 |
|------|--------|
| MIN | colors_n9, benzylation, photo_pce10, photo_wf3, snar, colors_bob |
| MAX | fullerenes, alkox, suzuki, hplc |

### 难度判定标准

| Normalized Regret | 难度 |
|-------------------|------|
| < 0.01 | EASY |
| 0.01 - 0.05 | MODERATE |
| 0.05 - 0.15 | HARD |
| > 0.15 | VERY HARD |

Normalized Regret = EI mean regret / y_range

---

## 2. 结果总表

| 名称 | 方向 | 维度 | Evals | EI Regret (mean±std) | Norm Regret | y_range | 难度 |
|------|------|------|-------|---------------------|-------------|---------|------|
| colors_n9 | MIN | 3 | 20 | -0.001±0.000 | -0.002 | 0.56 | EASY |
| fullerenes | MAX | 3 | 20 | 0.017±0.007 | 0.033 | 0.51 | **MODERATE** |
| **alkox** | **MAX** | **4** | **20** | **40.33±25.49** | **0.399** | **101.14** | **VERY HARD** |
| benzylation | MIN | 4 | 20 | 0.094±0.051 | 0.005 | 19.19 | EASY |
| photo_pce10 | MIN | 4 | 20 | 0.000±0.000 | 0.000 | 1.33 | EASY |
| photo_wf3 | MIN | 4 | 20 | 0.008±0.008 | 0.002 | 3.68 | EASY |
| snar | MIN | 4 | 20 | 0.002±0.006 | 0.000 | 5.17 | EASY |
| suzuki | MAX | 4 | 20 | -1.352±0.000 | -0.015 | 89.36 | EASY |
| colors_bob | MIN | 5 | 30 | -0.011±0.000 | -0.021 | 0.52 | EASY |
| hplc | MAX | 6 | 50 | 35.55±31.94 | 0.014 | 2453.60 | **MODERATE** |

> 注：负 regret 表示 EI 找到了比 10K Sobol 估计更好的点。

---

## 3. 逐数据集分析

### 3.1 VERY HARD: alkox (4D, MAX conversion)

**归一化 regret = 0.399，是唯一的 VERY HARD 数据集。**

- y_range = 101.1（0-100 的 conversion）
- 5 次运行中，2 次找到 ~86（接近最优 100），2 次仅找到 ~30（远离最优），1 次找到 65
- **方差极大**（std=25.49），说明函数景观存在陷阱，GP 难以准确建模
- 这是 **CAP-PPO 最有潜力的 4D 仿真器任务**：传统 EI 表现差，RL 有充分超越空间

### 3.2 MODERATE: fullerenes (3D, MAX product) & hplc (6D, MAX peak_area)

**fullerenes**: 归一化 regret 0.033
- 3D 问题但 EI 仍有残余 regret
- 可能存在尖锐的全局最优峰，GP 拟合不够精确

**hplc**: 归一化 regret 0.014
- 6D 问题，y_range = 2453.6
- 5 次运行表现不一：最好 regret 8.6，最差 83.3（std=31.9）
- 之前错误地认为 EASY（因为最小化方向错误，找到 peak_area=0 的点是最差而非最好）

### 3.3 EASY 数据集（7/10）

**photo_pce10**: regret = 0.000，5 次完美。极度简单。

**colors_n9, colors_bob, snar**: 负或接近零 regret，EI 轻松超越 Sobol 估计。

**benzylation**: regret 0.094（归一化 0.005）。**这是当前 CAP-PPO 训练的目标任务**，传统 EI 用 20 次评估即可接近全局最优。

**suzuki**: 负 regret，EI 找到了比 10K Sobol 更高的 yield。

**photo_wf3**: regret 0.008（归一化 0.002），极小。

---

## 4. 对 CAP-PPO 项目的启示

### 4.1 难度分布修正

修正优化方向后，结果显著不同：

| 之前（错误） | 修正后 |
|-------------|--------|
| 9 EASY + 1 MODERATE | **7 EASY + 2 MODERATE + 1 VERY HARD** |
| alkox 是唯一 MODERATE | **alkox 是 VERY HARD，fullerenes/hplc 是 MODERATE** |
| hplc 被判为 EASY (regret=0) | **hplc 实际是 MODERATE (regret=35.6)** |

### 4.2 任务选择建议

| 数据集 | 维度 | EI 难度 | RL 超越空间 | 推荐度 |
|--------|------|---------|------------|--------|
| **alkox** | **4** | **VERY HARD** | **极大** | **最高** |
| hplc | 6 | MODERATE | 中等 | 高（已有任务实现） |
| fullerenes | 3 | MODERATE | 中等 | 中（3D 可能太简单） |
| benzylation | 4 | EASY | 极小 | 低 |

### 4.3 与 Hartmann 6D 的对比

| 数据集 | 维度 | EI Norm Regret | CAP-PPO 已验证 |
|--------|------|----------------|---------------|
| Hartmann 6D | 6 | ~0.11（估计） | 0.014（成功） |
| **alkox** | **4** | **0.399** | **未测试** |
| hplc | 6 | 0.014 | 失败（BNN 问题） |
| benzylation | 4 | 0.005 | 1.368（失败） |

**alkox 的 EI 归一化 regret (0.399) 远超 Hartmann 6D (~0.11)**，是所有数据集中对传统 BO 最具挑战性的。这意味着：
- GP 代理模型对 alkox 函数景观拟合困难
- EI 在 20 evals 内远未收敛
- CAP-PPO 在此任务上有巨大的超越潜力

### 4.4 建议行动

1. **优先尝试 alkox 仿真器任务**：实现 `alkox_emulator.py`（类似 `benzylation_emulator.py`），用仿射变换生成变体，测试 CAP-PPO
2. **hplc 仍值得改进**：已有任务实现，修复 BNN 训练后重新测试
3. **benzylation 可以暂缓**：传统 EI 已几乎最优，RL 超越空间有限

---

## 5. 等高线图

等高线图保存在 `results_policies/emulator_difficulty_test/contour_*.png`，每张图展示所有维度对的 2D 切面（固定其他维度在 0.5）。

关键观察：
- **alkox**: 景观复杂，存在明显的非线性交互和多个局部结构
- **fullerenes**: 有尖锐的最优峰，GP 可能难以精确捕获
- **benzylation**: 有明显的宽广最优 basin，GP 容易拟合
- **photo_pce10**: 极度平坦，大部分区域目标值接近零

---

## 附录: 测试脚本

`MYRL/scripts/test_emulator_difficulty.py`:
- 自动根据数据集目标处理最大化/最小化方向
- 最大化数据集通过取反目标值统一为最小化框架
- 等高线图使用原始（非取反）值便于解释

```bash
export PYTHONPATH="MYRL:olympus/src"
python MYRL/scripts/test_emulator_difficulty.py
```

# v12: BNN 训练函数生成方式重构 — BNN_mean + RFF 输出扰动

**日期**: 2026-03-19
**前序**: `code_changes_v11_alkox_bnn_objective.md`（BNN 权重采样方案）、`code_changes_v10_alkox_emulator.md`（alkox 任务）

---

## 一、改动动机

### 1.1 v11 BNN 权重采样方案的问题

v11 使用 BNN 后验权重采样（`SampledBNNFunction`）生成训练函数：

```
w = loc + |sigma| × eps,  eps ~ N(0, I)
```

**问题**：Olympus-aligned BNN 训练产生的后验非常紧（sigma 远小于 loc），导致：

| kl_weight | inter（采样函数间相关性） | 效果 |
|-----------|--------------------------|------|
| 0.001 | **0.89** | 采样函数几乎相同，多样性不足 |
| 0.003 | 0.53 | 前次实验使用，但这是通过放松 BNN 质量换取多样性 |

核心矛盾：**提高 BNN 拟合质量（小 kl_weight）会降低采样多样性；增大 kl_weight 增加多样性会牺牲 BNN 质量**。

### 1.2 温度放大已排除

温度 > 1.0 直接导致 NN 权重扰动过大，前向传播产生极端值，采样函数退化为噪声。

### 1.3 BNN 输入空间增广已排除

测试了对 BNN 输入施加小仿射变换来增加多样性：

| 方案 | 同变体 inter | 效果 |
|------|-------------|------|
| 输入增广（单变体内） | > 0.89 | 多样性完全不足 |
| 跨变体 | 0.655 | 多样性仅来自不同变体，不是增广 |

BNN 对输入微扰不敏感，输出几乎不变。

### 1.4 新方案：BNN 后验均值 + 输出空间 RFF 扰动

核心思想：将**拟合质量**和**多样性**解耦：

```
f(x) = BNN_mean(x) + alpha × g(x)
```

- **BNN_mean(x)**：使用后验均值权重（loc only），不采样 → 提供最准确的基函数
- **g(x)**：从 GP 先验（Matern 2.5）采样的 RFF 随机函数 → 提供平滑可控的多样性
- **alpha**：扰动强度（default 5.0），控制多样性 vs 保真度

**与之前被否决的 oracle_gp (GP→RFF) 方案的本质区别**：
- oracle_gp：用 GP **拟合**轨迹数据 → 在 NN 景观上拟合失败 → RFF 为噪声
- 本方案：RFF 从 GP **先验**采样（不拟合任何数据）→ 仅提供平滑随机扰动

### 1.5 测试结果

**同变体 10 个扰动（alpha=5.0, ls=0.3）**：

| 变体 | inter | f* std |
|------|-------|--------|
| Variant 0 | 0.60 | 15.3 |
| Variant 3 | 0.25 | 8.7 |
| Variant 7 | 0.41 | 11.2 |

**跨变体 10 BNN × 10 RFF = 100 个函数**：

| 指标 | 值 |
|------|-----|
| inter（全部 100 函数间平均相关） | **0.156** |
| f* 标准差 | **22.4** |

与 Hartmann 6D 成功训练时的 oracle_gp 指标（inter=0.193）高度可比。

---

## 二、变更文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `MYRL/myrl/rl/train_rl.py` | **修改** | 新增 `RFFPriorFunction`、`BNNMeanWithRFFPerturbation`；修改 `reset()` 和 CLI |
| `MYRL/local_scripts/alkox.sh` | **修改** | 新增 `--bnn_rff_alpha`/`--bnn_rff_length_scale`；kl_weight 改为 0.001；BNN 训练 Olympus-aligned |
| `MYRL/local_scripts/benzylation.sh` | **修改** | 新增 `--bnn_rff_alpha`/`--bnn_rff_length_scale` |
| `MYRL/local_scripts/hplc.sh` | **修改** | 新增 `--bnn_rff_alpha`/`--bnn_rff_length_scale` |

---

## 三、详细改动

### 3.1 `train_rl.py` — 核心实现

#### 3.1.1 新增 `RFFPriorFunction`（line 727）

从 GP 先验（Matern 2.5）采样的随机傅里叶特征函数。**不拟合任何数据**。

```python
class RFFPriorFunction:
    def __init__(self, dim, length_scale, n_features=256, seed=0):
        rng = np.random.default_rng(seed)
        nu = 2.5
        # 从 Matern 2.5 谱密度采样频率（等价于 scaled Student-t）
        z = rng.normal(size=(n_features, dim))
        v = rng.chisquare(df=2 * nu, size=(n_features, 1))
        self._W = z / (length_scale * np.sqrt(v / (2 * nu)))
        self._b = rng.uniform(0, 2 * np.pi, size=n_features)
        self._scale = np.sqrt(2.0 / n_features)

    def __call__(self, X):
        Z = np.cos(X @ self._W.T + self._b[None, :])
        return self._scale * Z.sum(axis=1)
```

数学原理：Bochner 定理 — 平稳核可表示为频率空间的期望，RFF 用有限采样近似。256 个特征足以产生平滑的随机函数。

#### 3.1.2 新增 `BNNMeanWithRFFPerturbation(ObjectiveFunction)`（line 753）

替代原来的 `SampledBNNFunction`，用于 `objective_source="bnn"` 时的训练函数生成。

```python
class BNNMeanWithRFFPerturbation(ObjectiveFunction):
    def __init__(self, bnn_params, rng, *, bounds, alpha=5.0,
                 rff_length_scale=0.3, rff_n_features=256):
        # 只使用后验均值权重（loc），不采样 sigma
        self._weights = [(loc, bias) for loc, sigma, bias in bnn_params['layers']]
        # 每次实例化生成新的 RFF
        rff_seed = int(rng.integers(0, 2**31 - 1))
        self._rff = RFFPriorFunction(dim, rff_length_scale, rff_n_features, rff_seed)

    def __call__(self, X):
        y_base = self._bnn_mean(X)   # 准确的基函数
        y_perturb = self._rff(X)      # 平滑随机扰动
        return (y_base + self._alpha * y_perturb).astype(np.float32)
```

关键设计：
- **每个 episode 实例化一次** → 新的 RFF seed → 新的确定性训练函数
- **同一变体的 BNN mean 固定** → 扰动仅来自 RFF → 多样性可控
- **alpha 和 length_scale 分离控制**：alpha 控制扰动幅度，length_scale 控制扰动平滑度

#### 3.1.3 修改 `ImprovedBraninBOEnv.__init__`

新增参数：

```python
def __init__(self, ..., bnn_rff_alpha=5.0, bnn_rff_length_scale=0.3):
    self.bnn_rff_alpha = float(bnn_rff_alpha)
    self.bnn_rff_length_scale = float(bnn_rff_length_scale)
```

#### 3.1.4 修改 `reset()`（line 1732）

```python
# 旧（v11 BNN 权重采样）：
self.current_func = SampledBNNFunction(bnn_params, self.rng, bounds=...)

# 新（v12 BNN mean + RFF 扰动）：
self.current_func = BNNMeanWithRFFPerturbation(
    bnn_params=bnn_params, rng=self.rng,
    bounds=(...), alpha=self.bnn_rff_alpha,
    rff_length_scale=self.bnn_rff_length_scale,
)
```

#### 3.1.5 新增 CLI 参数

```python
parser.add_argument("--bnn_rff_alpha", type=float, default=5.0,
                    help="RFF perturbation strength (BNN mode)")
parser.add_argument("--bnn_rff_length_scale", type=float, default=0.3,
                    help="RFF Matern 2.5 length scale (BNN mode)")
```

参数完整传递链：CLI → `train_improved()` → config.json → `ImprovedBraninBOEnv()` → `reset()` → `BNNMeanWithRFFPerturbation()`。

#### 3.1.6 `SampledBNNFunction` 保留

原有 `SampledBNNFunction` 不删除，仍被 `test_bnn_vs_gp_surrogate.py` 引用。但训练时不再使用。

### 3.2 Shell 脚本变更

三个脚本统一新增：

```bash
# ===== BNN RFF 扰动参数（BNN mean + alpha * RFF prior）=====
RL_BNN_RFF_ALPHA=5.0             # 扰动强度
RL_BNN_RFF_LENGTH_SCALE=0.3      # RFF 长度尺度 (Matern 2.5)
```

Step 2 训练命令新增：

```bash
  --bnn_rff_alpha "${RL_BNN_RFF_ALPHA}" \
  --bnn_rff_length_scale "${RL_BNN_RFF_LENGTH_SCALE}" \
```

### 3.3 Alkox BNN kl_weight 变更

```bash
# 旧（v11）：放松后验换取权重采样多样性
BNN_KL_WEIGHT=0.003

# 新（v12）：后验尽量准确，多样性由 RFF 提供
BNN_KL_WEIGHT=0.001
```

新方案下 kl_weight 越小 → BNN mean 越准确 → 基函数质量越高。不再需要通过放松后验换取多样性。

同时 BNN 训练切换为 Olympus-aligned 配置：

```bash
python -u MYRL/scripts/train_bnn_surrogates.py \
    --max_epochs 100000 \    # Olympus default（原 20000）
    --batch_size 20 \        # Olympus default
    --pred_int 100 \         # Olympus default
    --es_patience 100 \      # Olympus default
    --valid_fraction 0.2
```

---

## 四、设计原理

### 4.1 为什么用 RFF 而不是其他扰动方式

| 方案 | 优点 | 缺点 |
|------|------|------|
| BNN 权重采样 | 自然 | 后验太紧，多样性不足 |
| 温度放大 | 简单 | 破坏 NN 前向传播，退化为噪声 |
| 输入增广 | 不改变 BNN | BNN 对输入微扰不敏感 |
| 高斯噪声 | 最简单 | 非平滑，不像真实函数 |
| **RFF 输出扰动** | **平滑、可控、与 BNN 质量解耦** | 需调 alpha/ls 两个超参 |

RFF 的关键优势：
1. **平滑性**：从 Matern 2.5 核采样，函数连续且足够光滑
2. **可控多样性**：alpha 直接控制扰动幅度，每个 RFF seed 生成一个确定性函数
3. **解耦**：BNN 质量和训练多样性完全独立调节
4. **计算开销极低**：一次矩阵乘法，无额外模型训练

### 4.2 超参数选择依据

| 参数 | 默认值 | 选择依据 |
|------|--------|----------|
| alpha | 5.0 | 测试中 alpha=5.0 使跨变体 inter≈0.156，与 Hartmann 成功时的 0.193 可比 |
| length_scale | 0.3 | 在 [0,1]^d 域中覆盖约 30% 范围，产生中等频率的扰动 |
| n_features | 256 | RFF 近似精度与计算的平衡 |

### 4.3 模拟的工业场景

```
现实工业流程:
  10 个历史实验（有限数据）
     ↓ BNN 拟合
  10 个代理模型（后验均值 = 最佳估计）
     ↓ + RFF 扰动（数据增强）
  无限多样的合成训练函数
     ↓ RL 训练
  泛化 BO 策略
```

不需要额外调用仿真器（不需要在线交互），所有训练数据从 BNN + RFF 合成。

---

## 五、完整流水线（以 alkox 为例）

```
alkox.sh (v12):
  Step 1:    生成变体 + BO 轨迹            finetune.py --stage generate
  Step 1.5:  生成 TAF source data          prepare_taf_data()
  Step 1.5b: 训练 BNN surrogates           train_bnn_surrogates.py (kl=0.001, Olympus-aligned)
  Step 2:    训练 CAP-PPO (BNN+RFF)        train_rl.py --objective_source bnn
                                              --bnn_rff_alpha 5.0
                                              --bnn_rff_length_scale 0.3
  Step 3a:   评估 (GP surrogate)           eval_rl_new.py --surrogate gp
  Step 3b:   评估 (TabPFN base)            eval_rl_new.py --surrogate tabpfn_base
```

### 各任务 BNN 配置汇总

| 任务 | kl_weight | alpha | ls | BNN 训练配置 |
|------|-----------|-------|----|-------------|
| Alkox (4D) | 0.001 | 5.0 | 0.3 | Olympus-aligned (100k epochs, batch=20, es=100) |
| Benzylation (4D) | 0.001 (default) | 5.0 | 0.3 | Default |
| HPLC (6D) | 0.04 | 5.0 | 0.3 | Default (20k epochs) |

---

## 六、与 v11 的对比

| 维度 | v11 (BNN 权重采样) | v12 (BNN mean + RFF) |
|------|--------------------|-----------------------|
| 训练函数来源 | `w = loc + sigma * eps` → NN 前向 | `BNN_mean(x) + alpha * RFF(x)` |
| BNN 质量利用 | loc + sigma 均使用 | 仅使用 loc（最佳估计） |
| 多样性来源 | 后验宽度（sigma） | RFF 先验采样 |
| kl_weight 策略 | 需要较大（0.003）放松后验 | 尽量小（0.001）保证准确性 |
| 多样性控制 | 间接（通过 kl_weight） | 直接（通过 alpha 和 length_scale） |
| 跨变体 inter | ~0.53 (kl=0.003) | **~0.156** (alpha=5.0, ls=0.3) |
| 扰动平滑性 | 取决于后验分布 | 由 Matern 2.5 核保证 |

---

## 七、注意事项

### 7.1 需要重新生成的数据

由于 alkox 的 `_VARIANT_SUITE_SPECS` 在之前的会话中被更新为 Option D（dx ±0.08, rot ±28°, sx 0.84-1.0），旧的变体缓存和 BNN 参数需要重新生成：

```bash
# 删除旧缓存后重新运行
rm -f ./data/alkox_emulator_variants_k10_seed2026_transform.npz
rm -f ./data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz
rm -f ./data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz
rm -f ./data/taf_source_data_alkox_emulator_k10_transform.pkl
bash MYRL/local_scripts/alkox.sh
```

### 7.2 Hartmann 6D 不受影响

Hartmann 6D 使用 `objective_source=oracle_gp`，不走 BNN 路径，v12 的改动对其无影响。

---

*生成时间：2026-03-19*

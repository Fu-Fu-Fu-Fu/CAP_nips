# MYRL

本目录是从 `Branin_0105/src` 的方法代码剥离/封装得到的可复用项目实现（代码已写入本目录，不依赖 import 原工程）。

当前内置任务：
- `branin_family`：与你现有 Branin 变体族一致（ID/OOD suite、轨迹驱动微调、CAP-PPO 训练/评估）
- `goldstein_price`：Goldstein-Price（“Gold函数”）示例任务
- `goldstein_price_family`：Goldstein-Price 变体族（与 branin_family 同样的 dx/rotation/sx 变换与 ID/OOD suite）
- `hartmann_3d`：Hartmann-3D（定义域 [0,1]^3；全局最小值约 -3.86278）
- `hartmann_3d_family`：Hartmann-3D 变体族（在 [0,1]^3 上做 dx/scale/3D 旋转变换；同样提供 in_range/ood_level_1/2/3）

**训练**
- 微调 TabPFN（生成/加载缓存 + finetune）：
  - `python MYRL/scripts/finetune.py --task branin_family --stage all`
  - `python MYRL/scripts/finetune.py --task goldstein_price --stage all`
  - `python MYRL/scripts/finetune.py --task goldstein_price_family --stage all`
  - `python MYRL/scripts/finetune.py --task hartmann_3d --stage all`
  - `python MYRL/scripts/finetune.py --task hartmann_3d_family --stage all`
- 训练 CAP-PPO（PPO）：
  - `python MYRL/scripts/train_rl.py --task branin_family --objective_source oracle_gp --save_dir ./runs/ppo_branin`
  - `python MYRL/scripts/train_rl.py --task goldstein_price --objective_source direct --save_dir ./runs/ppo_gold`
  - `python MYRL/scripts/train_rl.py --task goldstein_price_family --objective_source oracle_gp --save_dir ./runs/ppo_gprice_family`
  - `python MYRL/scripts/train_rl.py --task hartmann_3d --objective_source direct --save_dir ./runs/ppo_hartmann3d`
  - `python MYRL/scripts/train_rl.py --task hartmann_3d_family --objective_source oracle_gp --save_dir ./runs/ppo_hartmann3d_family`

**测试/评估**
- PFN 网格评估（base vs finetuned vs GP）：
  - `python MYRL/scripts/eval_pfn.py --task branin_family`
  - `python MYRL/scripts/eval_pfn.py --task goldstein_price`
  - `python MYRL/scripts/eval_pfn.py --task goldstein_price_family`
- 多策略对比评估（产出结果 JSON + 图；可选保存轨迹 npz）：
  - `python MYRL/scripts/eval_rl_new.py --task branin_family --rl_model_path /path/to/ppo_best.pt`
  - `python MYRL/scripts/eval_rl_new.py --task goldstein_price --rl_model_path /path/to/ppo_best.pt --n_variants_per_group 1`
  - `python MYRL/scripts/eval_rl_new.py --task goldstein_price_family --rl_model_path /path/to/ppo_best.pt`
  - `python MYRL/scripts/eval_rl_new.py --task hartmann_3d --rl_model_path /path/to/ppo_best.pt --n_variants_per_group 1 --no-plot_trajectories`

**只画图（不重跑评估）**
- 从 `eval_rl_new.py` 保存的 `results_*.json` 复画 rank/regret：
  - `python MYRL/scripts/plot.py --results_json /path/to/results_gp_fair.json --methods CAP-PPO,EI --groups in_range`
- 从 `eval_pfn.py` 保存的 `pfn_eval_data.pkl` 里，按组随机抽样画等高线图（不重跑全量评估）：
  - `python MYRL/scripts/eval_pfn.py --plot_contours_from_pkl /path/to/pfn_eval_data.pkl --contour_k_per_group 3 --contour_groups in_range,ood_level_1 --contour_context_sizes 20 --contour_save_dir ./figs/pfn_contours`

**新增任务（任务1/任务2复用）**
1. 在 `MYRL/myrl/tasks/` 新建 `<your_task>.py`，实现 `TaskSpec`：`bounds/dim/evaluate_numpy/sample_train_variants/sample_eval_suite/optimal_value或estimate_global_min`
2. 在 `MYRL/myrl/tasks/builtin.py` 注册：`register_task(YourTask())`
3. 之后所有脚本直接 `--task your_task` 即可复用微调/训练/评估/画图流程。

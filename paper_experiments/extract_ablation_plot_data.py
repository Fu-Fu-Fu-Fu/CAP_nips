"""
Extract plot data from ablation pkl files and save as enhanced JSON.

For each task (hartmann6d, alkox) x ablation (wo_crossattn, wo_synthetic, wo_rl)
x surrogate (gp, tabpfn_base), extract the same format as
scale_sweep_plot_data_enhanced.json used in the main experiments.

Output: paper_experiments/ablation_plot_data/{task}/{ablation}/results_{surrogate}/
            scale_sweep_plot_data_enhanced.json
"""
import json
import os
import pickle
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
OUT_ROOT = os.path.join(REPO, "paper_experiments", "ablation_plot_data")

# Source pkl paths
ABLATION_PKLS = {
    "hartmann6d": {
        "wo_crossattn": {
            "gp": "paper_experiments/ablation/component/hartmann6d/wo_crossattn/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/component/hartmann6d/wo_crossattn/results_tabpfn_base/scale_sweep_data.pkl",
        },
        "wo_synthetic": {
            "gp": "paper_experiments/ablation/component/hartmann6d/wo_synthetic/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/component/hartmann6d/wo_synthetic/results_tabpfn_base/scale_sweep_data.pkl",
        },
        "wo_rl": {
            "gp": "paper_experiments/ablation/wo_rl/hartmann6d/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/wo_rl/hartmann6d/results_tabpfn_base/scale_sweep_data.pkl",
        },
    },
    "alkox": {
        "wo_crossattn": {
            "gp": "paper_experiments/ablation/component/alkox/wo_crossattn/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/component/alkox/wo_crossattn/results_tabpfn_base/scale_sweep_data.pkl",
        },
        "wo_synthetic": {
            "gp": "paper_experiments/ablation/component/alkox/wo_synthetic/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/component/alkox/wo_synthetic/results_tabpfn_base/scale_sweep_data.pkl",
        },
        "wo_rl": {
            "gp": "paper_experiments/ablation/wo_rl/alkox/results_gp/scale_sweep_data.pkl",
            "tabpfn_base": "paper_experiments/ablation/wo_rl/alkox/results_tabpfn_base/scale_sweep_data.pkl",
        },
    },
}


def compute_enhanced_json(pkl_data: dict) -> dict:
    """Convert raw pkl data into enhanced JSON format with normalized stats."""
    scales = pkl_data["scales"]
    method_names = pkl_data["method_names"]
    trajectories = pkl_data["trajectories"]
    results = pkl_data["results"]
    n_init = pkl_data["n_init"]
    max_steps = pkl_data["max_steps"]

    # x-axis: evaluation indices
    sample_scale = str(scales[0])
    sample_method = method_names[0]
    T = trajectories[sample_scale][sample_method].shape[2]
    x = list(range(T))

    # --- Simple final stats (raw regret, mean ± sem) ---
    simple_final_stats = {}
    for s in scales:
        sk = str(s)
        simple_final_stats[sk] = {}
        for m in method_names:
            arr = trajectories[sk][m]  # (n_variants, n_runs, T)
            final = arr[:, :, -1].flatten()  # all final regrets
            simple_final_stats[sk][m] = {
                "mean": float(np.mean(final)),
                "sem": float(np.std(final) / np.sqrt(len(final))),
            }

    # --- Simple trajectory summary (raw regret, median + q30/q70) ---
    simple_trajectory_summary = {}
    for s in scales:
        sk = str(s)
        simple_trajectory_summary[sk] = {}
        for m in method_names:
            arr = trajectories[sk][m]  # (n_variants, n_runs, T)
            flat = arr.reshape(-1, T)  # (n_variants * n_runs, T)
            simple_trajectory_summary[sk][m] = {
                "median": np.median(flat, axis=0).tolist(),
                "q30": np.percentile(flat, 30, axis=0).tolist(),
                "q70": np.percentile(flat, 70, axis=0).tolist(),
            }

    # --- Normalized regret ---
    # Normalize each (variant, run) trace by its initial regret (t=0)
    # initial_regret comes from the Random method at t=0
    normalized_final_stats = {}
    normalized_trajectory_summary = {}
    initial_regret_summary = {}

    for s in scales:
        sk = str(s)
        # Get initial regrets from Random (or first method if Random not present)
        if "Random" in trajectories[sk]:
            init_regrets = trajectories[sk]["Random"][:, :, 0]  # (n_variants, n_runs)
        else:
            # Fallback: use first method's t=0
            first_m = method_names[0]
            init_regrets = trajectories[sk][first_m][:, :, 0]

        init_regrets_safe = np.maximum(init_regrets, 1e-8)

        initial_regret_summary[sk] = {
            "mean": float(np.mean(init_regrets)),
            "sem": float(np.std(init_regrets) / np.sqrt(init_regrets.size)),
        }

        normalized_final_stats[sk] = {}
        normalized_trajectory_summary[sk] = {}

        for m in method_names:
            arr = trajectories[sk][m]  # (n_variants, n_runs, T)
            # Normalize: divide each trace by its init regret
            normed = arr / init_regrets_safe[:, :, None]

            final_normed = normed[:, :, -1].flatten()
            normalized_final_stats[sk][m] = {
                "mean": float(np.mean(final_normed)),
                "sem": float(np.std(final_normed) / np.sqrt(len(final_normed))),
            }

            flat_normed = normed.reshape(-1, T)
            normalized_trajectory_summary[sk][m] = {
                "median": np.median(flat_normed, axis=0).tolist(),
                "q30": np.percentile(flat_normed, 30, axis=0).tolist(),
                "q70": np.percentile(flat_normed, 70, axis=0).tolist(),
            }

    enhanced = {
        "task": pkl_data["task"],
        "surrogate": pkl_data["surrogate"],
        "model": pkl_data["model"],
        "scales": scales,
        "n_variants": pkl_data["n_variants"],
        "n_runs": pkl_data["n_runs"],
        "max_steps": max_steps,
        "n_init": n_init,
        "seed": pkl_data["seed"],
        "method_names": method_names,
        "specs": pkl_data.get("specs", {}),
        "x": x,
        "normalization": {
            "definition": "per-run normalized regret: regret_trace[v, run, t] / max(initial_regret[v, run], 1e-8)",
            "initial_regret_source": "trajectories[scale][Random][:, :, 0]",
        },
        "simple_final_stats": simple_final_stats,
        "simple_trajectory_summary": simple_trajectory_summary,
        "normalized_final_stats": normalized_final_stats,
        "normalized_trajectory_summary": normalized_trajectory_summary,
        "initial_regret_summary": initial_regret_summary,
    }
    return enhanced


def make_json_safe(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def main():
    os.chdir(REPO)
    total = 0

    for task, ablations in ABLATION_PKLS.items():
        for ablation, surrogates in ablations.items():
            for surrogate, pkl_path in surrogates.items():
                print(f"\n--- {task} / {ablation} / {surrogate} ---")
                if not os.path.exists(pkl_path):
                    print(f"  SKIP: {pkl_path} not found")
                    continue

                with open(pkl_path, "rb") as f:
                    pkl_data = pickle.load(f)

                # Convert scale keys: pkl may use float or str keys
                trajs = pkl_data["trajectories"]
                fixed_trajs = {}
                for k, v in trajs.items():
                    fixed_trajs[str(k)] = v
                pkl_data["trajectories"] = fixed_trajs

                if "results" in pkl_data:
                    fixed_res = {}
                    for k, v in pkl_data["results"].items():
                        fixed_res[str(k)] = v
                    pkl_data["results"] = fixed_res

                if "specs" in pkl_data:
                    fixed_specs = {}
                    for k, v in pkl_data["specs"].items():
                        fixed_specs[str(k)] = v
                    pkl_data["specs"] = fixed_specs

                enhanced = compute_enhanced_json(pkl_data)

                out_dir = os.path.join(OUT_ROOT, task, ablation, f"results_{surrogate}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "scale_sweep_plot_data_enhanced.json")

                with open(out_path, "w") as f:
                    json.dump(make_json_safe(enhanced), f, indent=2)

                print(f"  Saved: {out_path}")
                print(f"  Methods: {enhanced['method_names']}")
                total += 1

    print(f"\n=== Done: {total} files saved to {OUT_ROOT} ===")


if __name__ == "__main__":
    main()

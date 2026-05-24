from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

import numpy as np  # noqa: E402

import myrl.eval.eval_rl_new as eval_mod  # noqa: E402


def _split_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return items or None


def _resolve_regret_plot_modes(mode: str) -> List[str]:
    mode = str(mode)
    if mode == "both":
        return ["mean_bootstrap_95", "median_30_70"]
    return [mode]


def _normalize_method(name: str) -> str:
    name = str(name).strip()
    return eval_mod.CAP_PPO_NAME if name == eval_mod.LEGACY_RL_NAME else name


def main():
    parser = argparse.ArgumentParser(description="Re-plot eval_rl_new results from saved JSON (no re-run).")
    parser.add_argument("--results_json", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None, help="Override output directory (default: alongside JSON)")
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated group names to plot (default: all)")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods to plot (default: all)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps (default: inferred from data)")
    parser.add_argument("--filename_prefix", type=str, default="comparison_replot")
    parser.add_argument(
        "--regret_plot",
        type=str,
        choices=["mean_bootstrap_95", "median_30_70", "both"],
        default="mean_bootstrap_95",
        help="How to plot regret curves.",
    )
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results_json: Dict[str, Any] = json.load(f)

    groups = _split_csv(args.groups)
    methods = _split_csv(args.methods)

    out_dir = args.save_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.results_json)), "replots")
    os.makedirs(out_dir, exist_ok=True)

    # Support both the old JSON format (group->method->...) and the new format:
    # {"__meta__": ..., "groups": {group: {"__meta__":..., "methods": {...}}}}
    groups_blob: Optional[Dict[str, Any]] = None
    if isinstance(results_json, dict) and "groups" in results_json:
        groups_blob = results_json.get("groups", {})
        meta = results_json.get("__meta__", {}) or {}
        group_names = groups or list(groups_blob.keys())
        get_group_data = lambda g: groups_blob[g].get("methods", {})  # noqa: E731
        default_max_steps = meta.get("max_steps", None)
        results_groups = groups_blob
    else:
        group_names = groups or list(results_json.keys())
        get_group_data = lambda g: results_json[g]  # noqa: E731
        default_max_steps = None
        results_groups = results_json

    for group_name in group_names:
        if group_name not in results_groups:
            raise KeyError(f"Group not found in results: {group_name}")

        group_data: Dict[str, Any] = get_group_data(group_name)
        method_names_raw = methods or list(group_data.keys())
        method_names = [_normalize_method(m) for m in method_names_raw]

        unit_curves_by_method: Dict[str, np.ndarray] = {}
        inferred_max_steps = None
        for method in method_names:
            # Allow loading legacy results keyed by "RL" while displaying as "CAP-PPO".
            key = method
            if key not in group_data and method == eval_mod.CAP_PPO_NAME and eval_mod.LEGACY_RL_NAME in group_data:
                key = eval_mod.LEGACY_RL_NAME
            if key not in group_data:
                continue
            regrets_by_variant = group_data[key]["regrets_by_variant"]
            unit_curves = []
            for v_runs in regrets_by_variant:
                arr = np.asarray(v_runs, dtype=np.float64)  # (n_runs, T)
                unit_curves.append(arr.mean(axis=0))
            curves = np.asarray(unit_curves, dtype=np.float64)
            unit_curves_by_method[method] = curves
            if inferred_max_steps is None and curves.size > 0:
                inferred_max_steps = int(curves.shape[1] - 1)

        if not unit_curves_by_method:
            continue

        max_steps = int(args.max_steps) if args.max_steps is not None else int(default_max_steps or inferred_max_steps or 0)
        group_dir = os.path.join(out_dir, group_name)
        for rp in _resolve_regret_plot_modes(str(args.regret_plot)):
            suffix = "" if rp == "mean_bootstrap_95" else f"_{rp}"
            eval_mod._plot_rank_and_regret(
                unit_curves_by_method,
                max_steps=max_steps,
                save_dir=group_dir,
                filename_prefix=f"{args.filename_prefix}{suffix}",
                title=f"{group_name}",
                colors={},
                regret_plot=str(rp),
            )


if __name__ == "__main__":
    main()

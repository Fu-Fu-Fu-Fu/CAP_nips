"""
Truncate K=10 data to K=5 for ablation experiments.

Handles: variants npz, trajectories npz, BNN params npz (Alkox), TAF pkl.

Usage:
    python paper_experiments/ablation/scripts/generate_k5_data.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MYRL'))

import numpy as np
import pickle
from pathlib import Path

K_NEW = 5
DATA_DIR = Path("data")
OUT_DIR = Path("data")  # save alongside originals with _k5_ suffix


def truncate_variants(src: Path, dst: Path):
    """Truncate variants npz from K=10 to K=5."""
    d = np.load(src, allow_pickle=True)
    variants = d["variants"][:K_NEW]
    metadata = d["metadata"]
    np.savez(dst, variants=variants, metadata=metadata)
    print(f"  Variants: {src.name} -> {dst.name} ({len(variants)} variants)")


def truncate_trajectories(src: Path, dst: Path):
    """Truncate trajectories npz, keeping only trajectories for variant_idx < K_NEW."""
    d = np.load(src, allow_pickle=True)
    vi = d["variant_indices"]
    mask = vi < K_NEW
    np.savez(
        dst,
        variants=d["variants"][:K_NEW],
        X_trajs=d["X_trajs"][mask],
        y_trajs=d["y_trajs"][mask],
        variant_indices=d["variant_indices"][mask],
        variant_infos=d["variant_infos"][:K_NEW],
        metadata=d["metadata"],
    )
    n_kept = mask.sum()
    print(f"  Trajectories: {src.name} -> {dst.name} ({n_kept}/{len(vi)} trajs kept)")


def truncate_bnn_params(src: Path, dst: Path):
    """Truncate BNN params npz from K=10 to K=5."""
    d = np.load(src, allow_pickle=True)
    out = {"n_variants": np.array(K_NEW)}
    for i in range(K_NEW):
        # Copy all layer_i_* keys
        l = 0
        while f"layer_{i}_{l}_loc" in d:
            for suffix in ("loc", "sigma", "bias"):
                key = f"layer_{i}_{l}_{suffix}"
                out[key] = d[key]
            l += 1
        out[f"y_mean_{i}"] = d[f"y_mean_{i}"]
        out[f"y_std_{i}"] = d[f"y_std_{i}"]
    np.savez(dst, **out)
    print(f"  BNN params: {src.name} -> {dst.name} ({K_NEW} variants)")


def truncate_taf(src: Path, dst: Path):
    """Truncate TAF source data pkl from K=10 to K=5."""
    with open(src, "rb") as f:
        d = pickle.load(f)
    out = {
        "D": d["D"],
        "M": K_NEW,
    }
    for key in ("X", "Y", "kernel_lengthscale", "kernel_variance",
                "noise_variance", "use_prior_mean_function"):
        out[key] = d[key][:K_NEW]
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print(f"  TAF: {src.name} -> {dst.name} (M={K_NEW})")


def main():
    print(f"=== Truncating K=10 data to K={K_NEW} ===\n")

    # --- Hartmann-6D ---
    print("Hartmann-6D:")
    truncate_variants(
        DATA_DIR / "hartmann_6d_family_variants_k10_seed2026.npz",
        OUT_DIR / "hartmann_6d_family_variants_k5_seed2026.npz",
    )
    truncate_trajectories(
        DATA_DIR / "hartmann_6d_family_bo_trajs_k10_boSeed2026.npz",
        OUT_DIR / "hartmann_6d_family_bo_trajs_k5_boSeed2026.npz",
    )
    truncate_taf(
        DATA_DIR / "taf_source_data_hartmann_6d_family_k10.pkl",
        OUT_DIR / "taf_source_data_hartmann_6d_family_k5.pkl",
    )

    # --- Alkox ---
    print("\nAlkox:")
    truncate_variants(
        DATA_DIR / "alkox_emulator_variants_k10_seed2026_transform.npz",
        OUT_DIR / "alkox_emulator_variants_k5_seed2026_transform.npz",
    )
    truncate_trajectories(
        DATA_DIR / "alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz",
        OUT_DIR / "alkox_emulator_bo_trajs_k5_boSeed2026_transform.npz",
    )
    truncate_bnn_params(
        DATA_DIR / "bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz",
        OUT_DIR / "bnn_surrogates_alkox_emulator_k5_kl0.001_transform.npz",
    )
    truncate_taf(
        DATA_DIR / "taf_source_data_alkox_emulator_k10_transform.pkl",
        OUT_DIR / "taf_source_data_alkox_emulator_k5_transform.pkl",
    )

    print("\nDone. All K=5 data saved to", OUT_DIR)


if __name__ == "__main__":
    main()

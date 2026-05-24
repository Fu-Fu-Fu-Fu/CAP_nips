from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

"""
Train BNN surrogates from BO trajectory data.

Architecture and training aligned with Olympus BayesNeuralNet:
  - DenseLocalReparameterization layers
  - leaky_relu(alpha=0.2)
  - NLL loss with learned aleatoric scale: -sum(Normal(pred, scale).log_prob(y))
  - KL normalized by batch_size, weighted by --kl_weight (Olympus 'reg', default 0.001)
  - Random batch sampling with replacement (Olympus _generator)
  - Early stopping on validation RMSD (Olympus convention)

Usage:
    python MYRL/scripts/train_bnn_surrogates.py \
        --trajs_path ./data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz \
        --output_path ./data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz
"""

import os
import argparse
import numpy as np


def build_and_train_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_nodes: int = 48,
    hidden_depth: int = 3,
    max_epochs: int = 100000,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.001,
    batch_size: int = 20,
    pred_int: int = 100,
    es_patience: int = 100,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Train a single BNN on (X_train, y_train), aligned with Olympus BayesNeuralNet.

    Returns dict with:
        layers: list of (loc, sigma, bias) for each dense layer
        y_mean: float
        y_std: float
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Standardize y
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    X = X_train.astype(np.float32)

    n_samples = X.shape[0]

    # --- Train / validation split ---
    n_valid = max(1, int(n_samples * valid_fraction))
    n_train = n_samples - n_valid
    perm = np.random.permutation(n_samples)
    train_idx, valid_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = X[train_idx], y_norm[train_idx]
    X_va, y_va = X[valid_idx], y_norm[valid_idx]

    # --- Build BNN (Olympus architecture) ---
    # leaky_relu with alpha=0.2, matching Olympus act_funcs
    def leaky_relu_02(y):
        return tf.nn.leaky_relu(y, alpha=0.2)

    layers = []
    for i in range(hidden_depth):
        layers.append(
            tfp.layers.DenseLocalReparameterization(
                hidden_nodes, activation=leaky_relu_02, name=f"dense_{i}",
            )
        )
    layers.append(
        tfp.layers.DenseLocalReparameterization(
            1, activation=None, name=f"dense_{hidden_depth}",
        )
    )
    model = tf.keras.Sequential(layers)
    _ = model(X_tr[:1])  # build

    # Learned aleatoric scale (Olympus: self.scale = softplus(Variable))
    scale_raw = tf.Variable(tf.zeros([1, 1]), name="scale_raw", dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    all_trainable = list(model.trainable_variables) + [scale_raw]

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            scale = tf.nn.softplus(scale_raw) + 1e-6  # positive aleatoric noise
            y_dist = tfd.Normal(loc=y_pred, scale=scale)
            # NLL with reduce_sum — Olympus convention
            reg_loss = -tf.reduce_sum(y_dist.log_prob(y_batch))
            # KL normalized by batch size — Olympus convention
            kl = tf.add_n(model.losses) / tf.cast(tf.shape(x_batch)[0], tf.float32)
            loss = reg_loss + kl_weight * kl
        grads = tape.gradient(loss, all_trainable)
        # Sanitize NaN/Inf gradients from reparameterization sampling
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, all_trainable))
        return loss

    def predict_mean(x, num_samples=10):
        """MC prediction: average over num_samples forward passes."""
        preds = np.stack(
            [model(x, training=False).numpy() for _ in range(num_samples)],
            axis=0,
        )
        return np.mean(preds, axis=0)  # (n, 1)

    # --- Training loop (Olympus convention) ---
    best_valid_rmsd = float("inf")
    patience_counter = 0
    diverged = False

    for epoch in range(1, max_epochs + 1):
        # Random batch with replacement — Olympus _generator
        batch_idx = np.random.randint(0, n_train, size=min(batch_size, n_train))
        x_batch = X_tr[batch_idx]
        y_batch = y_tr[batch_idx].reshape(-1, 1)

        loss_val = train_step(x_batch, y_batch)

        if not np.isfinite(float(loss_val)):
            print(f"    WARNING: loss became NaN/Inf at epoch {epoch}, aborting")
            diverged = True
            break

        if epoch % pred_int == 0:
            # Validate on held-out set — Olympus early stopping on valid RMSD
            valid_pred = predict_mean(X_va, num_samples=10)
            valid_rmsd = float(np.sqrt(np.mean(
                (valid_pred.reshape(-1) - y_va.reshape(-1)) ** 2
            )))

            if valid_rmsd < best_valid_rmsd:
                best_valid_rmsd = valid_rmsd
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= es_patience:
                print(
                    f"    Early stop at epoch {epoch} "
                    f"(patience={es_patience}, best_rmsd={best_valid_rmsd:.6f})"
                )
                break

            if epoch % (pred_int * 10) == 0:
                print(
                    f"    Epoch {epoch}: valid_rmsd={valid_rmsd:.6f}, "
                    f"best={best_valid_rmsd:.6f}, "
                    f"patience={patience_counter}/{es_patience}"
                )

    # --- Extract variational parameters ---
    layers_params = []
    for layer in model.layers:
        kernel_posterior = layer.kernel_posterior
        loc = kernel_posterior.distribution.loc.numpy()
        sigma = kernel_posterior.distribution.scale.numpy()  # positive, post-softplus

        bias_posterior = layer.bias_posterior
        bias_loc = bias_posterior.distribution.loc.numpy()

        layers_params.append((loc, sigma, bias_loc))

    has_nan = diverged or any(
        np.any(np.isnan(loc)) or np.any(np.isnan(sigma)) or np.any(np.isnan(bias))
        for loc, sigma, bias in layers_params
    )
    return {
        "layers": layers_params,
        "y_mean": y_mean,
        "y_std": y_std,
        "has_nan": has_nan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train BNN surrogates from BO trajectories (Olympus-aligned)"
    )
    parser.add_argument("--trajs_path", type=str, required=True,
                        help="Path to BO trajectories .npz")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .npz path for BNN params")
    parser.add_argument("--hidden_nodes", type=int, default=48,
                        help="Hidden layer width (Olympus default: 48)")
    parser.add_argument("--hidden_depth", type=int, default=3,
                        help="Number of hidden layers (Olympus default: 3)")
    parser.add_argument("--max_epochs", type=int, default=100000,
                        help="Max training epochs (Olympus default: 100000)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=0.001,
                        help="KL regularization weight (Olympus 'reg' default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size (Olympus default: 20)")
    parser.add_argument("--pred_int", type=int, default=100,
                        help="Validation frequency in epochs (Olympus default: 100)")
    parser.add_argument("--es_patience", type=int, default=100,
                        help="Early stopping patience (Olympus default: 100)")
    parser.add_argument("--valid_fraction", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print(f"Loading trajectories from {args.trajs_path}")
    data = np.load(args.trajs_path, allow_pickle=True)

    for k in ("X_trajs", "y_trajs", "variant_indices"):
        if k not in data:
            raise KeyError(
                f"Invalid trajectories cache (missing '{k}'): {args.trajs_path}"
            )

    X_trajs = np.asarray(data["X_trajs"])
    y_trajs = np.asarray(data["y_trajs"])
    variant_indices = np.asarray(data["variant_indices"], dtype=np.int64)

    variant_ids = np.sort(np.unique(variant_indices))
    n_variants = len(variant_ids)
    print(f"Found {n_variants} variants")
    print(f"Config: hidden={args.hidden_nodes}x{args.hidden_depth}, "
          f"kl_weight={args.kl_weight}, max_epochs={args.max_epochs}, "
          f"batch_size={args.batch_size}, pred_int={args.pred_int}, "
          f"es_patience={args.es_patience}, valid_frac={args.valid_fraction}")

    save_dict = {"n_variants": n_variants}

    for i, v_id in enumerate(variant_ids):
        print(f"\n--- Variant {i}/{n_variants} (id={v_id}) ---")

        idxs = np.where(variant_indices == v_id)[0]
        traj_idx = int(idxs.min())
        X_train = np.asarray(X_trajs[traj_idx], dtype=np.float32)
        y_train = np.asarray(y_trajs[traj_idx], dtype=np.float32).reshape(-1)

        print(f"  Data: X={X_train.shape}, y range=[{y_train.min():.4f}, {y_train.max():.4f}]")

        max_retries = 6
        params = None
        for attempt in range(max_retries):
            cur_seed = args.seed + i + attempt * 1000
            # Progressively reduce learning rate on retries
            cur_lr = args.learning_rate * (0.5 ** attempt)
            if attempt > 0:
                print(f"  Retry {attempt}/{max_retries} with seed={cur_seed}, lr={cur_lr:.1e}")
            params = build_and_train_bnn(
                X_train, y_train,
                hidden_nodes=args.hidden_nodes,
                hidden_depth=args.hidden_depth,
                max_epochs=args.max_epochs,
                learning_rate=cur_lr,
                kl_weight=args.kl_weight,
                batch_size=args.batch_size,
                pred_int=args.pred_int,
                es_patience=args.es_patience,
                valid_fraction=args.valid_fraction,
                seed=cur_seed,
            )
            if not params["has_nan"]:
                break
            print(f"  WARNING: NaN detected in variant {i} (attempt {attempt+1})")
        if params["has_nan"]:
            raise RuntimeError(
                f"Variant {i} failed to train after {max_retries} attempts (NaN in params)"
            )

        for l, (loc, sigma, bias) in enumerate(params["layers"]):
            save_dict[f"layer_{i}_{l}_loc"] = loc
            save_dict[f"layer_{i}_{l}_sigma"] = sigma
            save_dict[f"layer_{i}_{l}_bias"] = bias
            cv = np.median(np.abs(sigma) / (np.abs(loc) + 1e-8))
            print(f"  Layer {l}: shape={loc.shape}, CV_median={cv:.3f}")

        save_dict[f"y_mean_{i}"] = params["y_mean"]
        save_dict[f"y_std_{i}"] = params["y_std"]
        print(f"  y_mean={params['y_mean']:.4f}, y_std={params['y_std']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(args.output_path, **save_dict)
    print(f"\nSaved BNN surrogates to {args.output_path}")
    print(f"Keys: {list(save_dict.keys())[:10]}...")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from types import SimpleNamespace
from scipy.stats import norm
import pickle as pkl
import GPy
from sklearn import mixture

class UCB():
    def __init__(self, feature_order, kappa, D=None, delta=None):
        self.feature_order = feature_order
        self.kappa = kappa
        self.D = D
        self.delta = delta
        assert not (self.kappa == "gp_ucb" and self.D is None)
        assert not (self.kappa == "gp_ucb" and self.delta is None)

    def act(self, state, rng=None):
        state = state.numpy()
        ucbs = self.af(state)
        chooser = rng if rng is not None else np.random
        action = chooser.choice(np.flatnonzero(ucbs == ucbs.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        if self.kappa == "gp_ucb":
            timestep_idx = self.feature_order.index("timestep")
            timesteps = state[:, timestep_idx] + 1  # MetaBO timesteps start at 0
        else:
            timesteps = None

        kappa = self.compute_kappa(timesteps)
        # Minimization: use LCB = mean - kappa * std; we still argmax, so return -LCB.
        return -means + kappa * stds

    def compute_kappa(self, timesteps):
        # https: // arxiv.org / pdf / 0912.3995.pdf
        # https: // arxiv.org / pdf / 1012.2599.pdf
        if self.kappa == "gp_ucb":
            assert timesteps is not None
            nu = 1
            tau_t = 2 * np.log(timesteps ** (self.D / 2 + 2) * np.pi ** 2 / (3 * self.delta))
            kappa = np.sqrt(nu * tau_t)
        else:
            assert timesteps is None
            kappa = self.kappa
        return kappa

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class EI():
    def __init__(self, feature_order):
        self.feature_order = feature_order

    def act(self, state, rng=None):
        state = state.numpy()
        eis = self.af(state)
        chooser = rng if rng is not None else np.random
        action = chooser.choice(np.flatnonzero(eis == eis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        # Minimization EI: E[max(0, y_best - Y)]
        mask = stds != 0.0
        eis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        imps = incumbents[mask] - means[mask]
        zs[mask] = imps / stds[mask]
        pdf_zs = norm.pdf(zs)
        cdf_zs = norm.cdf(zs)
        eis[mask] = imps * cdf_zs[mask] + stds[mask] * pdf_zs[mask]
        eis[eis < 0.0] = 0.0
        return eis

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class PI():
    def __init__(self, feature_order, xi):
        self.feature_order = feature_order
        self.xi = xi

    def act(self, state, rng=None):
        state = state.numpy()
        pis = self.af(state)
        chooser = rng if rng is not None else np.random
        action = chooser.choice(np.flatnonzero(pis == pis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        # Minimization PI: P(Y <= y_best - xi)
        mask = stds != 0.0
        pis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        zs[mask] = (incumbents[mask] - self.xi - means[mask]) / stds[mask]
        cdf_zs = norm.cdf(zs)
        pis[mask] = cdf_zs[mask]
        return pis

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class TAF():
    # implements the Transfer Acquisition Function from Wistuba et. al., Mach Learn (2018)
    # https://rd.springer.com/content/pdf/10.1007%2Fs10994-017-5684-y.pdf
    def __init__(self, datafile, mode="me", rho=None):
        self.datafile = datafile
        self.models_source = []  # will be filled in self.generate_source_models()
        self.generate_source_models()
        self.mode = mode
        if self.mode == "me":
            if rho is not None:
                raise ValueError("TAF mode='me' does not use rho; please leave rho=None.")
            self.rho = None
        elif self.mode == "ranking":
            if rho is None:
                rho = 1.0
            if rho <= 0:
                raise ValueError("TAF mode='ranking' requires rho > 0.")
            self.rho = float(rho)
        else:
            raise ValueError("Unknown TAF-mode!")

    def generate_source_models(self):
        with open(self.datafile, "rb") as f:
            data = pkl.load(f)
        self.data = data

        self.D = data["D"]
        self.M = data["M"]
        for i in range(self.M):
            self.models_source.append(self.train_gp(X=data["X"][i], Y=data["Y"][i],
                                                    kernel_lengthscale=data["kernel_lengthscale"][i],
                                                    kernel_variance=data["kernel_variance"][i],
                                                    noise_variance=data["noise_variance"][i],
                                                    use_prior_mean_function=data["use_prior_mean_function"][i]))

    def act(self, state, X_target, model_target):
        state = state.numpy()
        tafs = self.af(state, X_target, model_target)
        action = np.random.choice(np.flatnonzero(tafs == tafs.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def train_gp(self, X, Y, kernel_lengthscale, kernel_variance, noise_variance, use_prior_mean_function):
        kernel = GPy.kern.RBF(input_dim=self.D,
                              variance=kernel_variance,
                              lengthscale=kernel_lengthscale,
                              ARD=True)

        if use_prior_mean_function:
            mf = GPy.core.Mapping(self.D, 1)
            mf.f = lambda X: np.mean(Y, axis=0)[0] if Y is not None else 0.0
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0
        else:
            mf = None

        normalizer = False

        gp = GPy.models.gp_regression.GPRegression(X, Y,
                                                   noise_var=noise_variance,
                                                   kernel=kernel,
                                                   mean_function=mf,
                                                   normalizer=normalizer)
        gp.Gaussian_noise.variance = noise_variance
        gp.rbf.lengthscale = kernel_lengthscale
        gp.rbf.variance = kernel_variance

        return gp

    def af(self, state, X_target, model_target):
        # gather predictions of target gp
        mean_idx = 0
        means_target = state[:, mean_idx]
        std_idx = 1
        stds_target = state[:, std_idx]
        incumbent_idx = std_idx + self.D + 1
        incumbents_target = state[:, incumbent_idx]

        # gather predicitions of source gps
        xs = state[:, std_idx + 1:std_idx + 1 + self.D]
        means_source, stds_source = [], []
        for i in range(self.M):
            cur_means, cur_vars = self.models_source[i].predict_noiseless(xs)
            cur_stds = np.sqrt(cur_vars)
            means_source.append(cur_means)
            stds_source.append(cur_stds)
        means_source = np.concatenate(means_source, axis=1)
        stds_source = np.concatenate(stds_source, axis=1)

        # compute weights
        if self.mode == "me":  # product of experts
            beta = 1 / (self.M + 1)
            weights = [beta * stds_source[:, i] ** (-2) for i in range(self.M)]
            weights.append(beta * stds_target ** (-2))
            weights = np.array(weights).T
        elif self.mode == "ranking":  # ranking-based
            t = X_target.shape[0] if X_target is not None else 0

            # Epanechnikov quadratic kernel
            def kern(a, b, rho):
                def gamma(x):
                    gamma = 3 / 4 * (1 - x ** 2) if x <= 1 else 0.0
                    return gamma

                kern = gamma(np.linalg.norm(a - b) / rho)
                return kern

            if model_target is None:
                raise ValueError("TAF mode='ranking' requires a non-None model_target with predict_noiseless().")

            # compute ranking-based meta-features (vectorized; avoids O(t^2) model predictions)
            chi = [np.zeros((t ** 2,)) for _ in range(self.M + 1)]
            if t >= 2:
                denom = float(t * (t - 1))
                scale = 1.0 / denom

                # predict means on X_target for each model (M source + 1 target)
                means = []
                for k in range(self.M):
                    mu_k, _ = self.models_source[k].predict_noiseless(X_target)
                    means.append(np.asarray(mu_k, dtype=np.float64).reshape(-1))
                mu_t, _ = model_target.predict_noiseless(X_target)
                means.append(np.asarray(mu_t, dtype=np.float64).reshape(-1))

                for k, mu in enumerate(means):
                    # Minimization: smaller mean => better rank
                    comp = (mu[:, None] < mu[None, :]).astype(np.float64) * scale
                    chi[k] = comp.reshape(-1)

            # compute weights
            weights = []
            for i in range(self.M + 1):
                weights.append(kern(chi[i], chi[self.M + 1 - 1], self.rho))

            weights = np.array(weights)
            weights = np.tile(weights, (xs.shape[0], 1))

        # compute EI(x) of target model
        mask = stds_target != 0.0
        eis_target, zs = np.zeros((means_target.shape[0],)), np.zeros((means_target.shape[0],))
        imps = incumbents_target[mask] - means_target[mask]
        zs[mask] = imps / stds_target[mask]
        pdf_zs = norm.pdf(zs)
        cdf_zs = norm.cdf(zs)
        eis_target[mask] = imps * cdf_zs[mask] + stds_target[mask] * pdf_zs[mask]
        eis_target[eis_target < 0.0] = 0.0

        # compute predicted improvements of source models
        incumbents_source = []
        for i in range(self.M):
            if X_target is None:
                cur_incumbent = incumbents_target[0]
            else:
                cur_incumbent = np.min(self.models_source[i].predict_noiseless(X_target)[0])
            incumbents_source.append(cur_incumbent)
        incumbents_source = np.array(incumbents_source)
        Is_source = incumbents_source - means_source
        Is_source[Is_source < 0.0] = 0.0

        # compute TAF
        source_af = np.sum((weights[:, :-1] * Is_source), axis=1)
        target_af = weights[:, -1] * eis_target
        weight_sum = np.sum(weights, axis=1)
        taf = (source_af + target_af) / weight_sum

        return taf

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass

class RandomPolicy():
    """Random baseline policy"""
    def __init__(self):
        pass

    def act(self, state, rng=None):
        """Randomly select a candidate"""
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        n_candidates = state.shape[0]
        chooser = rng if rng is not None else np.random
        if hasattr(chooser, "integers"):
            action = chooser.integers(0, n_candidates)
        else:
            action = chooser.randint(0, n_candidates)
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class FunBO():
    """FunBO OOD-Bench acquisition function (Aglietti et al., ICML 2025).

    Derived from Figure 3 of the paper.  Essentially EI with an
    exploration bonus controlled by beta:
        z = (f_best - mu + beta * sigma) / sigma
        AF = (f_best - mu + beta * sigma) * Phi(z) + sigma * phi(z)
    When beta=0 this reduces exactly to standard EI (minimisation form).
    """

    def __init__(self, feature_order, beta=1.0):
        self.feature_order = feature_order
        self.beta = float(beta)

    def act(self, state, rng=None):
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        vals = self.af(state)
        chooser = rng if rng is not None else np.random
        action = chooser.choice(np.flatnonzero(vals == vals.max()))
        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([0.0])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        mask = stds > 0.0
        vals = np.zeros(means.shape[0])
        imp = incumbents[mask] - means[mask] + self.beta * stds[mask]
        z = np.zeros_like(imp)
        z = imp / stds[mask]
        vals[mask] = imp * norm.cdf(z) + stds[mask] * norm.pdf(z)
        vals[vals < 0.0] = 0.0
        return vals

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class TuRBOPolicy():
    """TuRBO-1 baseline (Eriksson et al., NeurIPS 2019).

    Trust Region Bayesian Optimization: maintains an adaptive trust region
    centered at the current best point, fits its own GP (botorch SingleTaskGP),
    and selects the next point via Thompson Sampling within the trust region.

    This policy runs its own complete BO loop and does NOT use the framework's
    surrogate or candidate generation.  The dispatch in run_bo_with_policy
    short-circuits to ``run_full_bo()``.
    """

    def __init__(
        self,
        n_cand: int = 5000,
        length_init: float = 0.8,
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int | None = None,
        device: str = "cpu",
    ):
        self.n_cand = int(n_cand)
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.success_tolerance = int(success_tolerance)
        self._failure_tolerance_override = failure_tolerance
        self.device = device

    # ---- interface stubs (unused; TuRBO bypasses act()) ----
    def act(self, state, rng=None):
        raise RuntimeError("TuRBOPolicy.act() should not be called directly; "
                           "use run_full_bo() instead.")

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass

    # ---- core TuRBO-1 loop ----
    def run_full_bo(
        self,
        func,
        global_min: float,
        X_init: np.ndarray,
        y_init: np.ndarray,
        max_steps: int,
        rng: np.random.Generator = None,
        return_trajectory: bool = False,
        return_trace: bool = False,
    ):
        """Run full TuRBO-1 optimisation loop.

        Returns the same formats as ``run_bo_with_policy``:
        - default: list of regrets
        - return_trajectory=True: (regrets, trajectory_X)
        - return_trace=True: trace dict
        """
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from torch.quasirandom import SobolEngine

        if rng is None:
            rng = np.random.default_rng()

        lower, upper = func.bounds
        dim = len(lower)
        span = upper - lower
        span = np.where(span < 1e-12, 1.0, span)

        # ---- state ----
        X = X_init.copy()
        y = y_init.copy()
        best_val = float(y.min())

        length = self.length_init
        fail_tol = (self._failure_tolerance_override
                    if self._failure_tolerance_override is not None
                    else max(4, dim))
        succ_count = 0
        fail_count = 0

        # ---- book-keeping ----
        regrets = [max(float(y.min() - global_min), 0.0)]
        trajectory = [X.copy()] if return_trajectory else None
        selected_xs = [] if return_trace else None
        selected_ys = [] if return_trace else None
        best_y_trace = [float(y.min())] if return_trace else None

        sobol = SobolEngine(dim, scramble=True,
                            seed=int(rng.integers(0, 2**31)))

        for step in range(max_steps):
            # -- normalise data to [0,1]^d --
            X_01 = (X - lower) / span
            y_tensor = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
            X_tensor = torch.tensor(X_01, dtype=torch.float64)

            # -- fit GP --
            try:
                gp = SingleTaskGP(X_tensor, y_tensor)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
            except Exception:
                # If GP fitting fails, fall back to random
                x_new = rng.uniform(lower, upper, size=(1, dim)).astype(np.float32)
                y_new = func(x_new)[0]
                X = np.vstack([X, x_new])
                y = np.concatenate([y, [y_new]])
                regrets.append(max(float(y.min() - global_min), 0.0))
                if trajectory is not None:
                    trajectory.append(x_new.copy())
                if return_trace:
                    selected_xs.append(x_new.copy())
                    selected_ys.append(float(y_new))
                    best_y_trace.append(float(y.min()))
                continue

            # -- generate candidates in trust region [center ± length/2] --
            best_idx = int(np.argmin(y))
            center = X_01[best_idx]

            cands_01 = sobol.draw(self.n_cand).to(dtype=torch.float64)
            # Scale to trust region
            lb_tr = torch.clamp(torch.tensor(center) - length / 2, 0.0, 1.0)
            ub_tr = torch.clamp(torch.tensor(center) + length / 2, 0.0, 1.0)
            cands_01 = lb_tr + (ub_tr - lb_tr) * cands_01

            # -- Thompson sampling --
            gp.eval()
            with torch.no_grad():
                posterior = gp.posterior(cands_01)
                samples = posterior.rsample(torch.Size([1])).squeeze(0).squeeze(-1)
            pick = int(torch.argmin(samples).item())

            # -- map back to original space --
            x_new_01 = cands_01[pick].numpy()
            x_new = (x_new_01 * span + lower).astype(np.float32).reshape(1, -1)
            y_new = func(x_new)[0]

            # -- update trust region --
            if y_new < best_val:
                succ_count += 1
                fail_count = 0
                best_val = float(y_new)
            else:
                fail_count += 1
                succ_count = 0

            if succ_count >= self.success_tolerance:
                length = min(2.0 * length, self.length_max)
                succ_count = 0
            if fail_count >= fail_tol:
                length = length / 2.0
                fail_count = 0

            # -- restart if trust region collapsed --
            if length < self.length_min:
                length = self.length_init
                succ_count = 0
                fail_count = 0
                # Reset Sobol for fresh exploration
                sobol = SobolEngine(dim, scramble=True,
                                    seed=int(rng.integers(0, 2**31)))

            # -- update data --
            X = np.vstack([X, x_new])
            y = np.concatenate([y, [y_new]])

            regrets.append(max(float(y.min() - global_min), 0.0))
            if trajectory is not None:
                trajectory.append(x_new.copy())
            if return_trace:
                selected_xs.append(x_new.copy())
                selected_ys.append(float(y_new))
                best_y_trace.append(float(y.min()))

        # ---- return in the same format as run_bo_with_policy ----
        if return_trace:
            X_selected = (
                np.vstack(selected_xs).astype(np.float32)
                if selected_xs else np.empty((0, dim), dtype=np.float32)
            )
            y_selected = (
                np.asarray(selected_ys, dtype=np.float32)
                if selected_ys else np.empty((0,), dtype=np.float32)
            )
            return {
                "X_init": np.asarray(X_init, dtype=np.float32),
                "y_init": np.asarray(y_init, dtype=np.float32).reshape(-1),
                "X_selected": X_selected,
                "y_selected": y_selected,
                "best_y_trace": np.asarray(best_y_trace, dtype=np.float32),
                "regret_trace": np.asarray(regrets, dtype=np.float32),
                "X_all": np.asarray(X, dtype=np.float32),
                "y_all": np.asarray(y, dtype=np.float32).reshape(-1),
                "global_min": float(global_min),
            }
        if return_trajectory:
            return regrets, np.vstack(trajectory)
        return regrets


class PFNs4BOPolicy():
    """PFNs4BO baseline (Müller et al., ICML 2023).

    Uses a pretrained PFN transformer as surrogate + EI acquisition
    evaluated on the provided candidate set.  The PFN replaces both
    the surrogate model and the AF computation — it directly outputs
    EI values for each candidate.

    Inputs must be normalised to [0,1]^d before calling act().
    The policy stores raw context via set_context() and normalises
    internally.
    """

    def __init__(self, device="cpu", model_name="hebo_plus_model"):
        import pfns4bo
        from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

        model_path = pfns4bo.model_dict[model_name]
        self._model = torch.load(model_path, map_location=device, weights_only=False)
        self._model.eval()
        self._bo = TransformerBOMethod(self._model, device=device)
        self._device = device

        # Context (set per-step by run_bo_with_policy)
        self.X_context = None
        self.y_context = None
        self.X_candidates = None
        self.bounds = None

    def set_context(self, X_context, y_context, X_candidates, bounds):
        self.X_context = X_context
        self.y_context = y_context
        self.X_candidates = X_candidates
        self.bounds = bounds

    def act(self, state, rng=None):
        if self.X_context is None:
            raise RuntimeError("Must call set_context() before act() for PFNs4BOPolicy")

        lower, upper = self.bounds
        span = upper - lower
        span = np.where(span < 1e-12, 1.0, span)

        # Normalise X to [0,1]
        X_obs_01 = torch.tensor((self.X_context - lower) / span, dtype=torch.float32)
        X_cand_01 = torch.tensor((self.X_candidates - lower) / span, dtype=torch.float32)
        # PFN is trained to maximise; our framework minimises → negate y
        y_obs = torch.tensor(-self.y_context, dtype=torch.float32)

        with torch.no_grad():
            idx, acq_vals = self._bo.observe_and_suggest(
                X_obs_01, y_obs, X_cand_01, return_actual_ei=True,
            )

        action = torch.tensor([idx], dtype=torch.int64)
        value = torch.tensor([0.0])
        return action.squeeze(0), value.squeeze(0)

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        self.X_context = None
        self.y_context = None
        self.X_candidates = None
        self.bounds = None


class TAFMeHelper:
    def __init__(self, taf_data_path: str, max_steps: int):
        self.taf_policy = TAF(taf_data_path, mode="me")
        self.max_steps = int(max_steps)

    def _build_state_for_policies(
        self,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        y_context: np.ndarray,
        current_step: int,
    ) -> np.ndarray:
        incumbent = y_context.min()
        n_candidates = len(X_candidates)
        dim = X_candidates.shape[1]

        state = np.zeros((n_candidates, 2 + dim + 3), dtype=np.float32)
        state[:, 0] = pred_mean
        state[:, 1] = pred_std
        state[:, 2:2 + dim] = X_candidates
        state[:, 2 + dim] = incumbent
        state[:, 2 + dim + 1] = current_step
        state[:, 2 + dim + 2] = self.max_steps
        return state

    def compute(
        self,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        X_context: np.ndarray,
        y_context: np.ndarray,
        current_step: int,
    ) -> np.ndarray:
        state = self._build_state_for_policies(
            X_candidates, pred_mean, pred_std, y_context, current_step
        )
        taf_values = self.taf_policy.af(state, X_context, model_target=None)
        return np.asarray(taf_values, dtype=np.float32)


class RLPolicy():
    """
    CAP-PPO (Candidate Acquisition Policy trained with PPO) wrapper.
    Adapts the RL agent to the policies.py interface
    """
    def __init__(
        self,
        model_path,
        coord_dim=2,
        hidden_dim=128,
        n_self_attn_layers=3,
        n_cross_attn_layers=3,
        n_heads=8,
        max_steps=20,
        device="cpu",
        n_persistent_base=128,
        n_total_candidates=192,
        k_centers=2,
        local_h=1.5,
        local_h_decay=0.9,
        use_taf_feature=False,
    ):
        # Import here to avoid circular dependency
        from ..rl.train_rl import ImprovedDualTowerSelector

        self.device = device
        self.coord_dim = coord_dim
        self.max_steps = max_steps
        self.use_taf_feature = use_taf_feature

        # Persistent pool parameters
        self.n_persistent_base = int(n_persistent_base)
        self.n_total_candidates = int(n_total_candidates)
        self.k_centers = int(k_centers)
        self.local_h = float(local_h)
        self.local_h_decay = float(local_h_decay)

        # Load model weights (and infer training-time max_steps for step embedding)
        state_dict = torch.load(model_path, map_location=device)
        step_embed = state_dict.get("step_embed.weight", None)
        if step_embed is not None:
            policy_max_steps = int(step_embed.shape[0]) - 1
        else:
            policy_max_steps = max_steps

        # Auto-detect use_taf_feature from checkpoint if not explicitly set
        # Check candidate_embed input dimension: coord_dim+3 (no TAF) vs coord_dim+4 (with TAF)
        cand_embed_key = "candidate_embed.0.weight"
        if cand_embed_key in state_dict:
            cand_input_dim = state_dict[cand_embed_key].shape[1]
            self.use_taf_feature = (cand_input_dim == coord_dim + 4)

        # Initialize policy network with checkpoint-compatible max_steps
        self.policy_max_steps = policy_max_steps
        self.policy = ImprovedDualTowerSelector(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
            max_steps=policy_max_steps,
            use_taf_feature=self.use_taf_feature,
        ).to(device)

        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print(f"[CAP-PPO] Loaded model from {model_path} (use_taf_feature={self.use_taf_feature})")
        if self.max_steps > self.policy_max_steps:
            print(f"[CAP-PPO] Warning: eval max_steps={self.max_steps} > "
                  f"policy_max_steps={self.policy_max_steps}; clamping step embedding index.")

        # Store additional info needed for feature building
        self.X_context = None
        self.y_context = None
        self.X_candidates = None
        self.pred_mean = None
        self.pred_std = None
        self.bounds = None
        self.current_step = 0
        self.is_persistent = None
        self.taf_rank_norm = None

        # Persistent pool state (initialized per-episode via reset_persistent_pool)
        self.persistent_pool = None
        self.persistent_available = None

    def set_context(self, X_context, y_context, X_candidates, pred_mean, pred_std,
                   bounds, current_step, is_persistent=None, taf_rank_norm=None):
        """Set context information needed for RL feature building"""
        self.X_context = X_context
        self.y_context = y_context
        self.X_candidates = X_candidates
        self.pred_mean = pred_mean
        self.pred_std = pred_std
        self.bounds = bounds
        self.current_step = current_step
        self.is_persistent = is_persistent
        self.taf_rank_norm = taf_rank_norm

    def reset_persistent_pool(self, lower, upper, rng):
        """在每个评估 variant 开始时调用，生成持久 Sobol 池"""
        from scipy.stats.qmc import Sobol
        dim = len(lower)
        sobol = Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 100000)))
        self.persistent_pool = (sobol.random(self.n_persistent_base) * (upper - lower) + lower).astype(np.float32)
        self.persistent_available = np.ones(self.n_persistent_base, dtype=bool)

    def consume_persistent_point(self, action_idx, n_persistent_in_candidates):
        """标记被选中的持久点为已消耗"""
        if action_idx < n_persistent_in_candidates:
            original_indices = np.where(self.persistent_available)[0]
            self.persistent_available[original_indices[action_idx]] = False

    def act(self, state):
        """
        Select action using RL policy
        Note: state is not used directly; we use the context set via set_context()
        """
        if self.X_context is None:
            raise RuntimeError("Must call set_context() before act() for RLPolicy")

        # Build features in the format expected by ImprovedDualTowerSelector
        context_feat, candidate_feat = self._build_features()

        context_feat = context_feat.unsqueeze(0).to(self.device)
        candidate_feat = candidate_feat.unsqueeze(0).to(self.device)
        step_idx = min(int(self.current_step), int(self.policy_max_steps))
        step_tensor = torch.tensor([step_idx], device=self.device)

        with torch.no_grad():
            logits, value = self.policy(context_feat, candidate_feat, step_tensor)
            action = torch.argmax(logits, dim=-1).item()

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value.item()])
        return action.squeeze(0), value.squeeze(0)

    def _build_features(self):
        """
        Build features for ImprovedDualTowerSelector (与训练端一致)

        Context: [x_norm, y_rank]
        Candidate: [x_norm, μ_norm, σ_norm, is_persistent(, taf_rank_norm)]
        """
        lower, upper = self.bounds
        X_context = self.X_context
        y_context = self.y_context
        X_candidates = self.X_candidates
        pred_mean = self.pred_mean
        pred_std = self.pred_std

        # ========== Context features ==========
        X_ctx_norm = (X_context - lower) / (upper - lower + 1e-8)

        # Rank-based encoding: 0 = 最好, 1 = 最差
        n_ctx = len(y_context)
        if n_ctx > 1:
            ranks = np.argsort(np.argsort(y_context)).astype(np.float32)
            y_rank = ranks / (n_ctx - 1)
        else:
            y_rank = np.zeros(n_ctx, dtype=np.float32)

        context_feat = np.concatenate([
            X_ctx_norm,
            y_rank.reshape(-1, 1)
        ], axis=-1)

        # ========== Candidate features ==========
        X_cand_norm = (X_candidates - lower) / (upper - lower + 1e-8)

        y_best = y_context.min()
        y_range = float(y_context.max() - y_context.min())
        y_range = max(y_range, 1e-6)

        mean_norm = (pred_mean - y_best) / y_range
        std_norm = pred_std / y_range

        n_candidates = len(X_candidates)

        # is_persistent 标记
        if self.is_persistent is not None:
            is_persistent_arr = np.asarray(self.is_persistent, dtype=np.float32)
        else:
            is_persistent_arr = np.zeros(n_candidates, dtype=np.float32)

        parts = [
            X_cand_norm,
            mean_norm.reshape(-1, 1),
            std_norm.reshape(-1, 1),
            is_persistent_arr.reshape(-1, 1),
        ]

        # TAF ranking 特征
        if self.use_taf_feature:
            if self.taf_rank_norm is not None:
                parts.append(np.asarray(self.taf_rank_norm, dtype=np.float32).reshape(-1, 1))
            else:
                parts.append(np.zeros((n_candidates, 1), dtype=np.float32))

        candidate_feat = np.concatenate(parts, axis=-1)

        return (
            torch.FloatTensor(context_feat),
            torch.FloatTensor(candidate_feat)
        )


class MetaBOPolicy():
    """
    Wrapper for a trained MetaBO NeuralAF policy (PPO weights from MetaBO repo).

    This is optional and only used in eval_rl_new.py when --metabo_logpath is provided.
    """

    def __init__(
        self,
        *,
        logpath: str,
        load_iter: int,
        device: str = "cpu",
        n_features: int = 7,
    ):
        import os
        import sys
        import pickle as pkl

        # Add the sibling MetaBO repo to PYTHONPATH.
        # MYRL/myrl/policies -> MYRL -> test -> MetaBO
        metabo_repo_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "MetaBO")
        )
        if metabo_repo_dir not in sys.path:
            sys.path.insert(0, metabo_repo_dir)

        from metabo.policies.policies import NeuralAF  # type: ignore

        self.device = device
        self.logpath = str(logpath)
        self.load_iter = int(load_iter)
        self.n_features = int(n_features)

        params_path = os.path.join(self.logpath, f"params_{self.load_iter}")
        weights_path = os.path.join(self.logpath, f"weights_{self.load_iter}")

        with open(params_path, "rb") as f:
            params = pkl.load(f)
        policy_options = params.get("policy_options", None)
        if policy_options is None:
            raise ValueError(f"Missing policy_options in {params_path}")

        obs_space = SimpleNamespace(shape=(1, self.n_features))
        act_space = SimpleNamespace(n=1)
        self.pi = NeuralAF(obs_space, act_space, deterministic=True, options=policy_options).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device)
        self.pi.load_state_dict(state_dict)
        self.pi.eval()

    def act(self, state, rng=None):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))
        state = state.to(self.device)
        with torch.no_grad():
            action, value = self.pi.act(state)
        return action, value

    def set_requires_grad(self, flag):
        pass

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        self.X_context = None
        self.y_context = None
        self.X_candidates = None
        self.pred_mean = None
        self.pred_std = None
        self.bounds = None
        self.current_step = 0

# ---------------------------------------
# Import
# ---------------------------------------
 
import numpy as np
import random
import os
import sys
import pysindy as ps
# Add local PySINDy path so Python can find your development version
from pysindy.feature_library import WeightedWeakPDELibrary
import warnings 
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

data = np.load("data/double_pendulum_dataset.npz")
t = data['t']
Y_noisy = data["Y_noisy"]
Y_true = data["Y_true"]
sigmas = data["sigma"]
Y_hf = Y_noisy[sigmas==0.05]
Y_lf = Y_noisy[sigmas==0.5]

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
K = 100
deg = 5
threshold = 0.1
dt = t[1] - t[0]
train_ratios = np.linspace(0.01, 0.1, 10)   # from 1% to 10%
n_models = 100
deriv_method = ps.FiniteDifference(order=2)

# ---------------------------------------
# DATA PREPARATION
# ---------------------------------------
split = 100
Y_lf_sel = [y for y in Y_lf[:(split*10)]]
Y_hf_sel = [y for y in Y_hf[:split]]
Y_true_sel = [y for y in Y_true[split:]]

sample_weights = np.array([(1/s)**2 for s in sigmas])
sample_weights_lf = sample_weights[sigmas==0.5]
sample_weights_hf = sample_weights[sigmas==0.05]

# ---------------------------------------
# HELPER FUNCTION
# ---------------------------------------
def run_sindy_experiment(Y_train, Y_true, library, weights=None):
    """Train SINDy model and return mean R² on acceleration (last two components)."""

    # --- Weighted Weak PDE Library for training ---


    optimizer = ps.STLSQ(threshold=threshold)
    #     bagging=True,
    #     n_models=100,
    # )

    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(Y_train, t=dt, sample_weight=weights)

    # --- Build a standard library/optimizer pair for evaluation ---
    lib_eval = ps.PolynomialLibrary(degree=deg, include_bias=False)
    opt_eval = ps.STLSQ(threshold=threshold)
    model_eval = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval)
    model_eval.fit([Y_true[0]], t=dt)
    opt_eval.coef_ = optimizer.coef_

    # --- Evaluate on all trajectories ---
    # --- Evaluate on all trajectories ---
    from sklearn.metrics import r2_score

    r2_scores = []
    for Y_ref in Y_true:
        Ydot_true = deriv_method._differentiate(Y_ref, t=dt)
        Ydot_pred = model_eval.predict(Y_ref)
        # Evaluate only on acceleration components (last two)
        Ydot_true_acc = Ydot_true[:, -2:]
        Ydot_pred_acc = Ydot_pred[:, -2:]
        # Compute R² for this trajectory
        r2 = 1 - np.sum((Ydot_true_acc - Ydot_pred_acc)**2) / \
        np.sum((Ydot_true_acc - np.mean(Ydot_true_acc, axis=0))**2)
        r2_scores.append(r2)

    r2_mean = np.nanmean(r2_scores)
    return r2_mean


# ---------------------------------------
# EXPERIMENT
# ---------------------------------------
results = {r: {
    "lf_unweighted": [],
    "hf_unweighted": [],
    "combined_unweighted": [],
    "combined_weighted": []
} for r in train_ratios}

for ratio in train_ratios:
    print(f"\n=== Training Ratio {ratio:.4f} ===")
    library = ps.feature_library.WeightedWeakPDELibrary(
        ps.PolynomialLibrary(degree=deg, include_bias=False),
        spatiotemporal_weights=np.ones_like(t),
        spatiotemporal_grid=t,
        p=2, K=K
    )
    for i in range(n_models):
        # Random sampling indices
        idx_hf = np.random.choice(len(Y_hf_sel), int(ratio * len(Y_hf_sel)), replace=False)
        idx_lf = [10*i + j for i in idx_hf for j in range(10)]
        Y_lf_train = [Y_lf_sel[j] for j in idx_lf]
        Y_hf_train = [Y_hf_sel[j] for j in idx_hf]
        Y_comb_train = Y_lf_train + Y_hf_train

        w_lf = [sample_weights_lf[j] for j in idx_lf]
        w_hf = [sample_weights_hf[j] for j in idx_hf]
        w_comb = w_lf + w_hf

        # LF only
        score_lf = run_sindy_experiment(Y_lf_train, Y_true_sel, library)
        results[ratio]["lf_unweighted"].append(score_lf)

        # HF only
        score_hf = run_sindy_experiment(Y_hf_train, Y_true_sel, library)
        results[ratio]["hf_unweighted"].append(score_hf)

        # Combined (unweighted)
        score_comb_unw = run_sindy_experiment(Y_comb_train, Y_true_sel, library)
        results[ratio]["combined_unweighted"].append(score_comb_unw)

        # Combined (weighted)
        score_comb_w = run_sindy_experiment(Y_comb_train, Y_true_sel, library, weights=w_comb)
        results[ratio]["combined_weighted"].append(score_comb_w)

        if (i+1) % 1 == 0:
            print(f"  Run {i+1:03d}: "
                  f"LF={score_lf:.3f}, HF={score_hf:.3f}, "
                  f"Comb_unw={score_comb_unw:.3f}, Comb_w={score_comb_w:.3f}")

# ---------------------------------------
# AGGREGATE RESULTS
# ---------------------------------------
summary = {}
for ratio, scores in results.items():
    summary[ratio] = {
        k: (np.mean(v), np.std(v))
        for k, v in scores.items()
    }

print("\n=== SUMMARY (Mean ± Std of R²) ===")
for ratio, data in summary.items():
    print(f"\nTrain ratio {ratio:.2f}")
    for method, (m, s) in data.items():
        print(f"  {method:20s}: {m:.4f} ± {s:.4f}")

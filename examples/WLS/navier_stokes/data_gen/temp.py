#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pysindy as ps
import sys

from data_gen import *
from var_encoder_def import *

def fd_from_list(z_list, t_list, diff_method):
    dz_list = []
    for z, t in zip(z_list, t_list):
        dz = diff_method._differentiate(z, t)  # (Nt, d)
        dz_list.append(dz)
    return dz_list

def sindy_predict_derivatives(model_sindy, z_list):
    pred_list = []
    for z in z_list:
        pred_list.append(model_sindy.predict(z))  # (Nt, d)
    return pred_list


def residual_weights_from_derivatives(dz_true_list, dz_pred_list, 
                                      eps=1e-8, scheme="per_sample", clip=(1e-4, 1e4)):
    weights = []
    for dz_true, dz_pred in zip(dz_true_list, dz_pred_list):
        res = (dz_true - dz_pred)**2  # (Nt, d)
        if scheme == "per_sample":
            w = 1.0 / (res + eps)      # (Nt, d): different weight per component
        elif scheme == "per_traj":
            # One scalar per component for the whole trajectory
            var_j = np.mean(res, axis=0, keepdims=True)   # (1, d)
            w = 1.0 / (var_j + eps) * np.ones_like(res)   # (Nt, d)
        else:
            raise ValueError("Unknown scheme.")
        # Optional clipping for numerical stability
        if clip is not None:
            lo, hi = clip
            w = np.clip(w, lo, hi)
        weights.append(w)
    return weights

# ---------------------------------------------------------
# experiment grid
# ---------------------------------------------------------
noise_levels = [1.0]
r2_mf, r2_no, r2_ls = [], [], []

# fix time grid step (must match your encoder/training)
dt = 0.01

# use nonzero spatial and temporal correlation to avoid pure white noise
corr_len = 0.006   # ~13 px when N=128
phi = 0.9        # strong temporal AR(1)

for nl in tqdm(noise_levels):
    case = "noisy"
    smult = nl

    # IMPORTANT: set corr=corr_len and arr1=phi (you had corr=0.0)
    # Ensure data_gen.get_noisy_dataset forwards these into add_noise where:
    #   H = exp(-(KX^2+KY^2) * (corr_len*N)^2 / 2)
    # and avoid per-frame unit-variance normalization if you want visible smoothing.
    data_clean, data = get_noisy_dataset(seeds=range(5)
        # sadd=smult, smmult=0.25, arr1=phi, corr=corr_len, alpha=1, test=True, 
    )

    # -----------------------------------------------------
    # reshape to (Ntotal, Nx, Ny)
    # (Assuming data is (Nsims, Nx, Ny, Nt) as used below)
    # -----------------------------------------------------
    Nsims, Nx, Ny, Nt = data.shape

    X_all = data.transpose(0, 3, 1, 2).reshape(Nsims * Nt, Nx, Ny)
    X_all_clean = data_clean.transpose(0, 3, 1, 2).reshape(Nsims * Nt, Nx, Ny)
    
    mean, std = X_all.mean(dtype=np.float32), X_all.std(dtype=np.float32)
    mean_clean, std_clean = X_all_clean.mean(dtype=np.float32), X_all_clean.std(dtype=np.float32)

    # -----------------------------------------------------
    # load pretrained VAE
    # -----------------------------------------------------
    latent_dim = 4
    device = torch.device("cpu")                   
    model = ConvVAE(latent_dim=latent_dim, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(f"vae_model_fc{latent_dim}_{case}_noise{smult}_beta10.pt",
                                     map_location=device))

    # If not doing MC dropout, evaluate deterministically:
    n_mc = 1
    if n_mc == 1:
        model.eval()
    else:
        model.train()  # MC dropout active

    # -----------------------------------------------------
    # encoding helper
    # -----------------------------------------------------
    def encode_dataset(dataset, mean, std, Nsims, Nt, model, n_mc, device, latent_dim):
        z_array = np.zeros((Nsims, Nt, latent_dim), dtype=np.float32)
        var_array = np.zeros((Nsims, Nt, latent_dim), dtype=np.float32)
        t_array = np.arange(Nt) * dt

        for sim_idx in range(Nsims):
            om = dataset[sim_idx].transpose(2, 0, 1)  # (Nt, Nx, Ny)
            X_norm = (om - mean) / (std + 1e-12)
            X_tensor = torch.tensor(X_norm[:, None, :, :], dtype=torch.float32, device=device)

            with torch.no_grad():
                mu_accum = torch.zeros(Nt, latent_dim, device=device)
                mu_sq_accum = torch.zeros(Nt, latent_dim, device=device)
                var_accum = torch.zeros(Nt, latent_dim, device=device)

                for _ in range(n_mc):
                    mu, logvar = model.encode(X_tensor)
                    var = torch.exp(logvar)
                    mu_accum += mu
                    mu_sq_accum += mu * mu
                    var_accum += var

                mu_mean = (mu_accum / n_mc).cpu().numpy()
                epistemic_var = (mu_sq_accum / n_mc - (mu_accum / n_mc) ** 2).cpu().numpy()
                aleatoric_var = (var_accum / n_mc).cpu().numpy()
                total_var = epistemic_var + aleatoric_var

            z_array[sim_idx] = mu_mean
            var_array[sim_idx] = total_var

            # no CUDA cache to clear on CPU

        return z_array, t_array, var_array

    # -----------------------------------------------------
    # encode
    # -----------------------------------------------------
    z_noisy, t_noisy, var_noisy = encode_dataset(
        dataset=data, mean=mean, std=std,
        Nsims=Nsims, Nt=Nt, model=model, n_mc=n_mc,
        device=device, latent_dim=latent_dim
    )
    z_clean, t_clean, var_clean = encode_dataset(
        dataset=data_clean, mean=mean_clean, std=std_clean,
        Nsims=Nsims, Nt=Nt, model=model, n_mc=n_mc,
        device=device, latent_dim=latent_dim
    )

    # -----------------------------------------------------
    # SINDy set-up
    # -----------------------------------------------------
    threshold = 0.0025
    diff_method = ps.FiniteDifference()

    sindy_opt_nomf = ps.STLSQ(alpha=1e-12, threshold=threshold, normalize_columns=False)
    sindy_opt_mf = ps.STLSQ(alpha=1e-12, threshold=threshold, normalize_columns=False)
    sindy_opt_ls = ps.STLSQ(alpha=1e-12, threshold=threshold, normalize_columns=False)


    ode_lib = ps.PolynomialLibrary(degree=3, include_bias=False)

    sindy_nomf = ps.SINDy(
        optimizer=sindy_opt_nomf,
        feature_library=ode_lib,
        differentiation_method=diff_method
    )
    sindy_mf = ps.SINDy(
        optimizer=sindy_opt_mf,
        feature_library=ode_lib,
        differentiation_method=diff_method
    )
    sindy_ls = ps.SINDy(
        optimizer=sindy_opt_ls,
        feature_library=ode_lib,
        differentiation_method=diff_method
    )


    # -----------------------------------------------------
    # prepare lists (API expects lists per trajectory)
    # -----------------------------------------------------
    z_noisy_list = [z_noisy[i] for i in range(Nsims)]
    t_list = [t_noisy for _ in range(Nsims)]
    # CRITICAL FIX: pass weights PER TRAJECTORY, not a concatenated array
    weights_list = [1.0 / np.clip(var_noisy[i] + 1e-4,1e-4,1e4) for i in range(Nsims)]
    
    # Initial unweighted fit
    sindy_nomf.fit(z_noisy_list, t_list)
    sindy_mf.fit(z_noisy_list, t_list, sample_weight=weights_list)
    sindy_ls.fit(z_noisy_list, t_list)

    # IRLS iterations
    n_irls = 3
    for _ in range(n_irls):
        # 1) True and predicted derivatives
        dz_true_list = fd_from_list(z_noisy_list, t_list, diff_method)
        dz_pred_list = sindy_predict_derivatives(sindy_ls, z_noisy_list)

        # 2) Per-sample, per-component weights (Nt, d) for each trajectory
        weights_list = residual_weights_from_derivatives(
            dz_true_list, dz_pred_list,
            eps=1e-8, scheme="per_sample",
            clip=(1e-4, 1e4)
        )

        # Many PySINDy versions accept a list of weight arrays matching each trajectory.
        # Each array may be shape (Nt, d): weights per time step and per component.
        sindy_ls.fit(z_noisy_list, t_list, sample_weight=weights_list)

    # -----------------------------------------------------
    # evaluation across trajectories (not a fixed index=2)
    # -----------------------------------------------------
    from numpy import nan
    from sklearn.metrics import r2_score

    def eval_model(model_sindy, z_clean, start_idx=2500):
        r2_vals = []
        for i in range(min(Nsims, 5)):
            z0 = z_clean[i][start_idx:]           # (Nt_eval, d)
            if z0.shape[0] < 2:                    # guard
                r2_vals.append(nan)
                continue
            tspan = np.arange(z0.shape[0]) * dt
            z0_init = z0[0]
            try:
                z_sim = model_sindy.simulate(z0_init, tspan)
                r2_vals.append(r2_score(z0, z_sim))
            except Exception:
                r2_vals.append(nan)
        return np.nanmean(r2_vals)

    r2_mf_mean = eval_model(sindy_mf, z_clean, start_idx=2500)
    r2_no_mean = eval_model(sindy_nomf, z_clean, start_idx=2500)
    r2_ls_mean = eval_model(sindy_ls, z_clean, start_idx=2500)

    print(f"[noise={smult:0.3f}] R^2 (MF)={r2_mf_mean:0.4f} R^2 (no-MF)={r2_no_mean:0.4f} R^2 (LS)={r2_ls_mean:0.4f}")

    r2_mf.append(r2_mf_mean)
    r2_no.append(r2_no_mean)
    r2_ls.append(r2_ls_mean)
    
    del data, data_clean

# Optionally: store results
np.savez("sindy_r2_results.npz", noise=np.array(noise_levels), r2_mf=np.array(r2_mf), r2_no=np.array(r2_no), r2_ls=np.array(r2_ls))

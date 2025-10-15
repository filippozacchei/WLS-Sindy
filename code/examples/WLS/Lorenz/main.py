from pathlib import Path
from utils.plot import plot_heatmap
from utils.part1 import evaluate_mf_sindy
from generator import generate_lorenz_data
import numpy as np

if __name__ == "__main__":
    system_name = "lorenz"
    out_dir = "./Results"

    # Define grid and parameters
    n_lf_vals = np.arange(10, 101, 10)
    n_hf_vals = np.arange(1, 11, 1)
    runs = 25
    dt = 1e-3
    threshold = 0.5
    degree = 2

    # Ground-truth Lorenz model coefficients (for MAD)
    C_true = np.zeros((9, 3))
    C_true[0, 0] = -10.0     # dx/dt = -10x + 10y
    C_true[1, 0] = 10.0
    C_true[0, 1] = 28.0      # dy/dt = 28x - xz - y
    C_true[5, 1] = -1.0
    C_true[1, 1] = -1.0
    C_true[4, 2] = 1.0       # dz/dt = xy - (8/3)z
    C_true[2, 2] = -8.0 / 3.0

    # Run the unified evaluation routine
    evaluate_mf_sindy(
        generator=generate_lorenz_data,
        system_name=system_name,
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        runs=runs,
        dt=dt,
        threshold=threshold,
        degree=degree,
        out_dir=out_dir,
        C_true=C_true,
    )

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    data = np.load(Path(out_dir) / system_name / f"{system_name}_results.npz")

    # Example plots
    plot_heatmap(np.clip(data["mf_score"], 0, None), "MF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "scores_mf.png")
    plot_heatmap(np.clip(data["dlf_score"], None, 1), "MF−LF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "scores_mf_minus_lf.png")
    plot_heatmap(np.clip(data["dhf_score"], None, 1), "MF−HF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "scores_mf_minus_hf.png")

    plot_heatmap(data["mf_mad"], "MF MAD", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "mad_mf.png", label="MAD")
    plot_heatmap(data["dlf_mad"], "MF−LF MAD", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "mad_mf_minus_lf.png", label="ΔMAD")
    plot_heatmap(data["dhf_mad"], "MF−HF MAD", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "mad_mf_minus_hf.png", label="ΔMAD")

    plot_heatmap(data["mf_dis"], "MF Disagreement", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "dis_mf.png", label="Disagreement")
    plot_heatmap(data["dlf_dis"], "MF−LF Disagreement", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "dis_mf_minus_lf.png", label="ΔDisagreement")
    plot_heatmap(data["dhf_dis"], "MF−HF Disagreement", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir) / "dis_mf_minus_hf.png", label="ΔDisagreement")

    print(f"\n✅ Lorenz evaluation complete. Results and figures saved to {out_dir}")

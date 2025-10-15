from pathlib import Path
import numpy as np
from utils.plot import plot_heatmap

system_name = "lorenz"
out_dir = "./Results"

# Define grid and parameters
n_lf_vals = np.arange(10, 101, 10)
n_hf_vals = np.arange(1, 11, 1)
data = np.load(Path(out_dir) / system_name / f"{system_name}_results.npz")

# Example plots
plot_heatmap(np.clip(data["mf_score"], 0, None), "MF $R^2$", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf.png")
plot_heatmap(np.clip(data["lf_score"], None, 1), "LF $R^2$", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_lf.png")
plot_heatmap(np.clip(data["hf_score"], None, 1), "HF $R^2$", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_hf.png")
plot_heatmap(np.clip(data["dlf_score"], None, 1), "MF−LF $R^2$", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf_minus_lf.png")
plot_heatmap(np.clip(data["dhf_score"], None, 1), "MF−HF $R^2$", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "scores_mf_minus_hf.png")

plot_heatmap(data["mf_mad"], "MF MAD", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf.png", label="MAD")
plot_heatmap(data["dlf_mad"], "LF MAD", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_lf.png", label="ΔMAD")
plot_heatmap(data["dhf_mad"], "HF MAD", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_hf.png", label="ΔMAD")
plot_heatmap(data["dlf_mad"], "MF−LF MAD", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf_minus_lf.png", label="ΔMAD")
plot_heatmap(data["dhf_mad"], "MF−HF MAD", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "mad_mf_minus_hf.png", label="ΔMAD")

plot_heatmap(data["mf_dis"], "MF Disagreement", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf.png", label="Disagreement")
plot_heatmap(data["dlf_dis"], "LF Disagreement", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_lf.png", label="ΔDisagreement")
plot_heatmap(data["dhf_dis"], "HF Disagreement", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_hf.png", label="ΔDisagreement")
plot_heatmap(data["dlf_dis"], "MF−LF Disagreement", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf_minus_lf.png", label="ΔDisagreement")
plot_heatmap(data["dhf_dis"], "MF−HF Disagreement", n_lf_vals, n_hf_vals,
                fname=Path(out_dir) / "dis_mf_minus_hf.png", label="ΔDisagreement")

print(f"\nLorenz evaluation complete. Results and figures saved to {out_dir}")

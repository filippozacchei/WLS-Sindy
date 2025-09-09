import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp
from pysindy.utils import lorenz

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection is registered)
import numpy as np
from scipy.integrate import solve_ivp
# choose ONE backend you have installed:
# QtAgg (requires PyQt5 or PySide6) or TkAgg (uses Tkinter)
import matplotlib
matplotlib.use("TkAgg")  # or "TkAgg"
from matplotlib import pyplot as plt

def plot_training_data(data_configuration, data_dict, lf_noise, hf_noise, n_lf, n_hf, run_id,dt = 0.001, t_span = (0, 1)):

    max_n_hf, max_n_lf = max(data_configuration["n_trajectories_hf"]), max(data_configuration["n_trajectories_lf"])
    x_hf_full, t_hf_full = data_dict["hf"][hf_noise]
    x_lf_full, t_lf_full = data_dict["lf"][lf_noise]
    hf_offset, lf_offset = run_id * max_n_hf, run_id * max_n_lf

    x_train_hf, _ = x_hf_full[hf_offset:hf_offset+n_hf], t_hf_full[hf_offset:hf_offset+n_hf]
    x_train_lf, _ = x_lf_full[lf_offset:lf_offset+n_lf], t_lf_full[lf_offset:lf_offset+n_lf]

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, 2))

    solution = solve_ivp(
        fun=lorenz,
        t_span=(0, 50),
        y0=[-8, 8, 27],
        t_eval=np.arange(t_span[0], 50, dt)
    )
    x, y, z = solution.y
    ax.plot(x, y, z, color='lightgrey', lw=0.5, linestyle='-')


    for sol in x_train_lf:
        x, y, z = sol[:,0], sol[:,1], sol[:,2]
        ax.scatter(x, y, z, color=colors[0], linestyle=':',lw=1,marker='o',s=5,alpha=0.25)
        
    for sol in x_train_hf:
        print(sol.shape)
        x, y, z = sol[:,0], sol[:,1], sol[:,2]
        ax.plot(x, y, z, color=colors[1], linestyle=':',lw=3)

    ax.set_xlabel("x", fontsize=14, color='white')
    ax.set_ylabel("y", fontsize=14, color='white')
    ax.set_zlabel("z", fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.set_axis_off()
    plt.savefig('Figures/training_data_example.png', dpi=300, transparent=True)
    plt.show()
    
def plot_training_data_interactive(
    data_configuration, data_dict, lf_noise, hf_noise, n_lf, n_hf, run_id,
    dt=0.001, t_span=(0, 1), save_path="Figures/training_data_example.png", show=True
):
    # offsets
    max_n_hf = max(data_configuration["n_trajectories_hf"])
    max_n_lf = max(data_configuration["n_trajectories_lf"])
    x_hf_full, t_hf_full = data_dict["hf"][hf_noise]
    x_lf_full, t_lf_full = data_dict["lf"][lf_noise]
    hf_offset, lf_offset = run_id * max_n_hf, run_id * max_n_lf

    # select training slices
    x_train_hf, _ = x_hf_full[hf_offset:hf_offset + n_hf], t_hf_full[hf_offset:hf_offset + n_hf]
    x_train_lf, _ = x_lf_full[lf_offset:lf_offset + n_lf], t_lf_full[lf_offset:lf_offset + n_lf]

    # figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # reference trajectory (Lorenz)
    solution = solve_ivp(
        fun=lorenz,
        t_span=(0, 50),
        y0=[-8, 8, 27],
        t_eval=np.arange(t_span[0], 50, dt)
    )
    x, y, z = solution.y
    ax.plot(x, y, z, color='lightgrey', lw=0.5, linestyle='-')

    # LF (points) and HF (lines)
    cmap = plt.cm.jet(np.linspace(0, 1, 2))
    for sol in x_train_lf:
        ax.scatter(sol[:, 0], sol[:, 1], sol[:, 2],
                   color=cmap[0], marker='o', s=5, alpha=0.25)

    for sol in x_train_hf:
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2],
                color=cmap[1], linestyle=':', lw=2)

    # axes/labels
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    # keep axes visible for orientation during navigation
    # ax.set_axis_off()  # <- leave commented if you want ticks and box

    # save optionally
    if save_path:
        fig.savefig(save_path, dpi=300, transparent=True)

    # this opens a **window** with pan/zoom/rotate via mouse
    if show:
        plt.show(block=True)

    return fig, ax
"""
lorenz_generator.py

Dynamic generator for Lorenz system trajectories.
Supports low-, high-, or multi-fidelity data generation on the fly.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Lorenz equations
# ---------------------------------------------------------------------
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Standard Lorenz system ODEs."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# ---------------------------------------------------------------------
# Single trajectory generator
# ---------------------------------------------------------------------
def generate_lorenz_trajectory(
    y0=None,
    T=10.0,
    dt=1e-3,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    noise_level=0.0,
    seed=None,
):
    """
    Generate one Lorenz trajectory (possibly noisy).

    Parameters
    ----------
    y0 : array-like, shape (3,)
        Initial condition. Random if None.
    T : float
        Total integration time.
    dt : float
        Time step.
    sigma, rho, beta : float
        Lorenz parameters.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    t : ndarray (Nt,)
        Time vector.
    X : ndarray (Nt, 3)
        Trajectory (x, y, z) with optional noise.
    Xdot : ndarray (Nt, 3)
        Time derivatives.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)

    if y0 is None:
        y0 = rng.uniform([-10, -10, 20], [10, 10, 30])

    sol = solve_ivp(lorenz, (t[0], t[-1]), y0, t_eval=t,
                    args=(sigma, rho, beta),
                    method="LSODA", rtol=1e-10, atol=1e-12)
    
    X = sol.y.T
    Xdot = np.array([lorenz(ti, xi, sigma, rho, beta) for ti, xi in zip(sol.t, X)])
    
    if noise_level > 0:
        X += rng.normal(0, noise_level, size=X.shape)

    return t, X, Xdot


# ---------------------------------------------------------------------
# Multi-trajectory generator (for LF/HF/MF)
# ---------------------------------------------------------------------
def generate_lorenz_flows(
    n_traj=1,
    T=10.0,
    dt=1e-3,
    noise_level=0.0,
    fidelity="hf",
    seed=42,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
):
    """
    Generate multiple Lorenz trajectories, analogous to generate_compressible_flow.

    Parameters
    ----------
    n_traj : int
        Number of trajectories.
    T : float
        Total simulation time.
    dt : float
        Time step.
    noise_level : float
        Gaussian noise level.
    fidelity : str
        'lf', 'hf', or 'mf'.
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    trajectories, derivatives, times = [], [], []

    for i in range(n_traj):
        y0 = rng.uniform([-10, -10, 20], [10, 10, 30])
        t, X, Xdot = generate_lorenz_trajectory(
            y0=y0, T=T, dt=dt,
            sigma=sigma, rho=rho, beta=beta,
            noise_level=noise_level, seed=seed+i
        )
        trajectories.append(X)
        derivatives.append(Xdot)
        times.append(t)

    return trajectories, derivatives, times


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
def plot_lorenz_3d(X, title="Lorenz Attractor", ax=None):
    """3D trajectory plot."""
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()


def animate_lorenz(X, t, save_path="lorenz.gif"):
    """Animate Lorenz trajectory in 3D."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], lw=1.5)

    ax.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))
    ax.set_zlim(np.min(X[:, 2]), np.max(X[:, 2]))
    ax.set_title("Lorenz trajectory")

    def update(frame):
        line.set_data(X[:frame, 0], X[:frame, 1])
        line.set_3d_properties(X[:frame, 2])
        return line,

    anim = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)
    anim.save(save_path, writer="pillow", fps=30)
    plt.close(fig)


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    trajectories, derivatives, times = generate_lorenz_flows(
        n_traj=3, T=5.0, dt=1e-3, noise_level=0.05, seed=0
    )

    # Plot and animate first trajectory
    plot_lorenz_3d(trajectories[0])
    animate_lorenz(trajectories[0], times[0])

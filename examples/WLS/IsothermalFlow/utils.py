import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../src")  # if needed for your local sindy package
from sindy import eWSINDy
from matplotlib.animation import FuncAnimation

def compressible(t, U, dx, N, mu, RT):
    u = U.reshape(N, N, 3)[:, :, 0]
    v = U.reshape(N, N, 3)[:, :, 1]
    rho = U.reshape(N, N, 3)[:, :, 2]
    ux = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(u, dx)
    uy = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(u, dx)
    uxx = ps.differentiation.FiniteDifference(
        d=2,
        axis=0,
        periodic=True,
    )._differentiate(u, dx)
    uyy = ps.differentiation.FiniteDifference(
        d=2,
        axis=1,
        periodic=True,
    )._differentiate(u, dx)
    vx = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(v, dx)
    vy = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(v, dx)
    vxx = ps.differentiation.FiniteDifference(
        d=2,
        axis=0,
        periodic=True,
    )._differentiate(v, dx)
    vyy = ps.differentiation.FiniteDifference(
        d=2,
        axis=1,
        periodic=True,
    )._differentiate(v, dx)
    px = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(rho * RT, dx)
    py = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(rho * RT, dx)
    ret = np.zeros((N, N, 3))
    ret[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) / rho
    ret[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) / rho
    ret[:, :, 2] = -(u * px / RT + v * py / RT + rho * ux + rho * vy)
    
    return ret.reshape(3 * N * N)


N = 64
Nt = 1000
L = 5
T = 5
mu = 1
RT = 1

t = np.linspace(0, T, Nt)
x = np.arange(0, N) * L / N
y = np.arange(0, N) * L / N
dx = x[1] - x[0]

# some arbitrary initial conditions
X, Y = np.meshgrid(x, y, indexing="ij")
y0 = np.zeros((N, N, 3))
y0[:, :, 0] = -np.sin(2 * np.pi * X / L) * np.exp(-((X - L/2)**2 + (Y - L/2)**2))
y0[:, :, 1] =  np.cos(2 * np.pi * Y / L) * np.exp(-((X - L/2)**2 + (Y - L/2)**2))
y0[:, :, 2] = 1 + 0.2 * np.exp(-((X - L/2)**2 + (Y - L/2)**2))

# rng = np.random.default_rng(42)
# y0 = np.zeros((N, N, 3))
# y0[:, :, 0] = np.sin(2 * np.pi * X / L) + 0.1 * rng.standard_normal((N, N))
# y0[:, :, 1] = np.cos(2 * np.pi * Y / L) + 0.1 * rng.standard_normal((N, N))
# y0[:, :, 2] = 1 + 0.1 * rng.standard_normal((N, N))

# y0 = np.zeros((N, N, 3))
# y0[:, :, 0] = np.tanh((Y - L/2) / 0.1)  # shear in y
# y0[:, :, 1] = 0.05 * np.sin(2 * np.pi * X / L)  # small perturbation
# y0[:, :, 2] = 1.0

sol = solve_ivp(
    compressible,
    (t[0], t[-1]),
    y0=y0.reshape(3 * N * N),
    t_eval=t,
    args=(dx, N, mu, RT),
    method="RK45",
    rtol=1e-8,
    atol=1e-8,
)

u_shaped_noiseless = sol.y.reshape(N, N, 3, -1).transpose(0, 1, 3, 2)
u_dot_noiseless = ps.FiniteDifference(d=1, axis=2)._differentiate(u_shaped_noiseless, t)

# Assume you already have u_shaped_noiseless of shape (Nx, Ny, Nt, 3)
Nt = u_shaped_noiseless.shape[2]

# # Set up the figure and three axes
# fig, axes = plt.subplots(1, 3, figsize=(24, 10))
# titles = [r"$\dot{u}$", r"$\dot{v}$", r"$\rho$"]

# # Precompute clim limits for each component
# clims = [
#     (-np.max(np.abs(u_shaped_noiseless[:, :, :, 0])), np.max(np.abs(u_shaped_noiseless[:, :, :, 0]))),
#     (-np.max(np.abs(u_shaped_noiseless[:, :, :, 1])), np.max(np.abs(u_shaped_noiseless[:, :, :, 1]))),
#     (np.min(u_shaped_noiseless[:, :, :, 2]), np.max(u_shaped_noiseless[:, :, :, 2])),
# ]

# ims = []
# for ax, title, cidx, clim in zip(axes, titles, range(3), clims):
#     im = ax.imshow(u_shaped_noiseless[:, :, 0, cidx], cmap="jet", animated=True)
#     im.set_clim(*clim)
#     fig.colorbar(im, ax=ax, fraction=0.045)
#     ax.set_xlabel("x", fontsize=16)
#     ax.set_ylabel("y", fontsize=16)
#     ax.set_title(title, fontsize=16)
#     ims.append(im)

# plt.tight_layout()

# # Update function for animation
# def update(i):
#     for j, im in enumerate(ims):
#         im.set_array(u_shaped_noiseless[:, :, i, j])
#     return ims

# # Create animation
# ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=False)

# plt.show()


spatiotemporal_grid = np.zeros((N, N, Nt, 3))
spatiotemporal_grid[:, :, :, 0] = x[:, np.newaxis, np.newaxis]
spatiotemporal_grid[:, :, :, 1] = y[np.newaxis, :, np.newaxis]
spatiotemporal_grid[:, :, :, 2] = t[np.newaxis, np.newaxis, :]

library_functions = [lambda x: x, lambda x: 1 / (1e-6 + abs(x))]
library_function_names = [lambda x: x, lambda x: x + "^-1"]

print("Model fitting...")
model = eWSINDy(
    library_functions=ps.CustomLibrary(library_functions=library_functions,function_names=library_function_names),
    pde=True,
    spatiotemporal_grid=spatiotemporal_grid,
    derivative_order=2,                  # KS needs up to 4th derivatives
    K=2000,                               # number of test functions
    H_xt=[L / 10, L / 10, T / 10],       # test function support
)
model.fit(
    [u_shaped_noiseless],
    [t],
    sample_weight=None,
    threshold=0.5,
    alpha=1e-12,
    max_iter=100,
    sample_ensemble=True,
)
print(model.coef_median)
print(model.get_feature_names())

# import re

# def prettify_feature(name, state_names):
#     """
#     Convert raw feature name (e.g. 'x1x0_2') into math form (e.g. 'v * u_yy').
#     """
#     # Replace x0, x1, x2 with u, v, p
#     for i, s in enumerate(state_names):
#         name = name.replace(f"x{i}^-1", f"{s}^-1")   # inverse terms first
#         name = name.replace(f"x{i}", s)

#     # Replace derivative suffixes
#     def repl_deriv(match):
#         base = match.group(1)  # e.g. u
#         suf  = match.group(2)  # e.g. 22 or 1
#         mapping = {"1": "_x", "2": "_y",
#                    "11": "_xx", "22": "_yy",
#                    "12": "_xy", "21": "_xy"}
#         return base + mapping.get(suf, "_" + suf)
    
#     name = re.sub(r'([uvw p]+)_([0-9]+)', repl_deriv, name)
#     return name

# def print_sindy_system(coefs, poly, state_names, tol=1e-8):
#     raw_features = poly.get_feature_names()
#     features = [prettify_feature(f, state_names) for f in raw_features]

#     for j, state in enumerate(state_names):
#         terms = []
#         for name, coef in zip(features, coefs[:, j]):
#             if abs(coef) > tol:
#                 terms.append(f"{coef:.3e} * {name}")
#         rhs = " + ".join(terms) if terms else "0"
#         print(f"{state}_t = {rhs}")
        
# state_names = ["u", "v", "p"]   # you have 3 equations
# print_sindy_system(coefs_median, poly, state_names)
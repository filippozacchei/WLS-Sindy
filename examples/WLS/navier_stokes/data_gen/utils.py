"""
utils.py
------------------
2-D incompressible vorticity–streamfunction solver for generating training data
for SINDy/WSINDy.

Model:
    ∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω + f(x,y,t)
    ∇²ψ = -ω,  u = ∂ψ/∂y,  v = -∂ψ/∂x

Features:
- Spectral Poisson solver and derivatives (FFT-based, periodic BCs).
- Shear layer, vortex pair, and random initial conditions.
- Optional localized or streamwise forcing.
- Noise injection utilities for robustness testing.
"""
import numpy as np
import scipy.ndimage as nd

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional
from scipy.integrate import solve_ivp

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
# ============================================================
# Spectral grid and operators
# ============================================================

@dataclass(frozen=True)
class SpectralGrid:
    N: int
    L: float
    x: np.ndarray
    y: np.ndarray
    dx: float
    kx: np.ndarray
    ky: np.ndarray
    KX: np.ndarray
    KY: np.ndarray
    K2: np.ndarray
    ikx: np.ndarray
    iky: np.ndarray

def make_grid(N: int, L: float) -> SpectralGrid:
    x = np.arange(N) * (L / N)
    y = np.arange(N) * (L / N)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    return SpectralGrid(N, L, x, y, dx, k, k, KX, KY, K2, 1j*KX, 1j*KY)

def spectral_poisson_psi(omega: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    mask = grid.K2 != 0
    psi_hat[mask] = -omega_hat[mask] / grid.K2[mask]
    return np.fft.ifft2(psi_hat).real

def spectral_grad(field: np.ndarray, grid: SpectralGrid) -> Tuple[np.ndarray, np.ndarray]:
    F = np.fft.fft2(field)
    return np.fft.ifft2(grid.ikx * F).real, np.fft.ifft2(grid.iky * F).real

def spectral_laplacian(field: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    return np.fft.ifft2(-grid.K2 * np.fft.fft2(field)).real

def velocities_from_vorticity(omega: np.ndarray, grid: SpectralGrid) -> Tuple[np.ndarray, np.ndarray]:
    psi = spectral_poisson_psi(omega, grid)
    u = spectral_grad(psi, grid)[1]    # u = ∂ψ/∂y
    v = -spectral_grad(psi, grid)[0]   # v = -∂ψ/∂x
    return u, v


# ============================================================
# Vorticity dynamics
# ============================================================

@dataclass(frozen=True)
class VorticityConfig:
    N: int = 128
    L: float = 2*np.pi
    nu: float = 2e-3
    U0: float = 0.0
    forcing: Optional[dict] = None
    # NEW: optional downstream sponge to suppress periodic wrap-around
    sponge: Optional[dict] = None  # e.g., {"x0": 0.80, "width": 0.12, "rate": 1.0}

def add_noise_deform_multiscale(
    omega,
    fixed_seed=None,
    sigma_add=0.0,
    sigma_mult=0.0,
    corr_len=0.02,
    deform_strength=(0.05, 0.01),  # (coarse, fine)
    preserve_rms=True,
):
    """
    Apply multiscale deformation + additive/multiplicative noise.
    
    Parameters
    ----------
    omega : ndarray
        Input vorticity field (Nx, Ny, Nt).
    fixed_seed : int, optional
        For reproducibility.
    sigma_add : float
        Std for additive Gaussian noise.
    sigma_mult : float
        Std for multiplicative noise.
    corr_len : float
        Correlation length for generating smooth displacement.
    deform_strength : tuple of floats
        (coarse_strength, fine_strength) controlling warp amplitudes.
    preserve_rms : bool
        If True, rescale noisy field to preserve RMS of clean omega.
    """
    rng = np.random.default_rng(fixed_seed)
    Nx, Ny, Nt = omega.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    omega_noisy = np.empty_like(omega)

    for t in range(Nt):
        f = omega[:, :, t]

        # --- Coarse displacement (smooth) ---
        dx_coarse = rng.normal(0, 1, size=(Nx, Ny))
        dy_coarse = rng.normal(0, 1, size=(Nx, Ny))
        dx_coarse = nd.gaussian_filter(dx_coarse, sigma=Nx*corr_len)
        dy_coarse = nd.gaussian_filter(dy_coarse, sigma=Ny*corr_len)
        dx_coarse *= deform_strength[0]
        dy_coarse *= deform_strength[0]

        # --- Fine displacement (sharper features) ---
        dx_fine = rng.normal(0, 1, size=(Nx, Ny))
        dy_fine = rng.normal(0, 1, size=(Nx, Ny))
        dx_fine = nd.gaussian_filter(dx_fine, sigma=Nx*corr_len/4)
        dy_fine = nd.gaussian_filter(dy_fine, sigma=Ny*corr_len/4)
        dx_fine *= deform_strength[1]
        dy_fine *= deform_strength[1]

        # --- Combine displacements ---
        dx = dx_coarse + dx_fine
        dy = dy_coarse + dy_fine

        # Warp coordinates
        Xd = (X + dx).clip(0, 1) * (Nx-1)
        Yd = (Y + dy).clip(0, 1) * (Ny-1)

        # Interpolate deformed field
        f_def = nd.map_coordinates(f, [Xd.ravel(), Yd.ravel()],
                                   order=1, mode="reflect").reshape(Nx, Ny)

        # Additive + multiplicative noise
        f_noisy = f_def * (1.0 + sigma_mult * rng.standard_normal(f.shape)) \
                  + sigma_add * rng.standard_normal(f.shape)

        # RMS preservation (optional)
        if preserve_rms:
            rms_clean = np.sqrt(np.mean(f**2))
            rms_noisy = np.sqrt(np.mean(f_noisy**2))
            if rms_noisy > 1e-12:
                f_noisy *= rms_clean / rms_noisy

        omega_noisy[:, :, t] = f_noisy

    return omega_noisy

def vorticity_rhs(t, W, grid, cfg):
    omega = W.reshape(grid.N, grid.N)

    u, v = velocities_from_vorticity(omega, grid)
    omega_x, omega_y = spectral_grad(omega, grid)
    U0 = cfg.U0 if cfg.U0 is not None else 0.0

    adv  = (u + U0) * omega_x + v * omega_y
    diff = cfg.nu * spectral_laplacian(omega, grid)

    # forcing selection
    if cfg.forcing and cfg.forcing.get("mode") == "bluff_penalized":
        f = bluff_penalized_forcing(t, omega, grid, cfg)
    elif cfg.forcing and cfg.forcing.get("mode") == "karman":
        f = karman_shedding_forcing(t, grid, cfg)
    else:
        f = 0.0

    # downstream sponge (unchanged)
    sponge_term = 0.0
    if cfg.sponge is not None:
        x0_rel  = cfg.sponge.get("x0", 0.80)
        w_rel   = cfg.sponge.get("width", 0.12)
        rate    = cfg.sponge.get("rate", 1.0)
        x0 = x0_rel * grid.L
        w  = max(1e-12, w_rel * grid.L)
        X, _ = np.meshgrid(grid.x, grid.y, indexing="ij")
        S = 0.5 * (1 + np.tanh((X - x0)/w))
        sponge_term = -rate * S * omega

    return (-adv + diff + f + sponge_term).reshape(-1)

def cylinder_mask(grid, x_c, y_c, D, smooth=0.03):
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    R = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    r0 = 0.5 * D
    w  = max(1e-12, smooth * D)
    # inside≈1, outside≈0 (smooth Heaviside)
    return 0.5 * (1.0 - np.tanh((R - r0)/w))

def grad_mag_mask(chi, grid):
    F = np.fft.fft2(chi)
    chi_x = np.fft.ifft2(grid.ikx * F).real
    chi_y = np.fft.ifft2(grid.iky * F).real
    return np.sqrt(chi_x**2 + chi_y**2)

def bluff_penalized_forcing(t, omega, grid, cfg):
    p   = cfg.forcing
    x_c = p.get("x_c", 0.25*grid.L)
    y_c = p.get("y_c", 0.50*grid.L)
    D   = p.get("D",   0.10*grid.L)

    alpha    = p.get("alpha",   50.0)  # damping inside body
    annulus  = p.get("annulus", 0.04)  # shell thickness (×D)
    c_shell  = p.get("c_shell",  3.0)  # shear source gain
    seed_amp = p.get("seed_amp", 0.5)  # antisymmetry to trigger shedding

    chi = cylinder_mask(grid, x_c, y_c, D, smooth=annulus)
    g   = grad_mag_mask(chi, grid)

    # 1) Brinkman-type damping inside the body
    f_damp  = -alpha * chi * omega

    # 2) Shear generation on the annulus, proportional to |∇chi| and U0
    U0 = cfg.U0 if cfg.U0 is not None else 0.0
    f_shell =  c_shell * U0 * g * np.sign(np.meshgrid(grid.x, grid.y, indexing="ij")[1] - y_c)

    # 3) Tiny antisymmetric seed localized at the cylinder edge
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    R = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    shell = np.exp(-((R - 0.5*D)**2) / (2*(0.5*annulus*D)**2))
    f_seed = seed_amp * shell * np.sign(Y - y_c)

    # Mild high-k damping of the forcing to avoid ringing
    F = np.fft.fft2(f_damp + f_shell + f_seed)
    k = 2*np.pi*np.fft.fftfreq(grid.N, d=grid.dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    lowpass = np.exp(-0.5*((KX**2+KY**2)/(0.6*np.max(abs(k))**2)))
    return np.fft.ifft2(F * lowpass).real

# def karman_shedding_forcing(t: float, grid: SpectralGrid, cfg: VorticityConfig) -> np.ndarray:
#     """
#     Forcing term that emulates a solid cylinder (bluff body) by 
#     applying localized drag in the region of the body. This creates 
#     vorticity naturally as the background flow is slowed.
#     """

#     p = cfg.forcing
#     x_c   = p.get("x_c", 0.25*grid.L)
#     y_c   = p.get("y_c", 0.5*grid.L)
#     D     = p.get("D", 0.10*grid.L)      # cylinder diameter
#     A     = p.get("A", 50.0)             # damping amplitude
#     sigma = p.get("sigma", 0.3*D)        # smoothing around cylinder

#     X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
#     R2 = (X-x_c)**2 + (Y-y_c)**2
#     mask = np.exp(-R2/(2*sigma**2))      # smooth bump over the cylinder

#     # Apply drag-like vorticity forcing
#     # Sign chosen so that it *resists* flow
#     f = -A * mask
#     return f

# ============================================================
# Initial conditions
# ============================================================

def ic_random_vorticity(grid: SpectralGrid, rng: np.random.Generator, amp: float = 2) -> np.ndarray:
    noise = rng.standard_normal((grid.N, grid.N))
    F = np.fft.fft2(noise)
    filt = (np.sqrt(grid.K2) <= 0.25*np.max(np.abs(grid.kx))).astype(float)
    omega = amp * np.fft.ifft2(F*filt).real
    return omega - omega.mean()

# ============================================================
# Trajectory integration
# ============================================================

def integrate_trajectory(omega0: np.ndarray, grid: SpectralGrid, cfg: VorticityConfig,
                         t_span: Tuple[float,float], Nt: int,
                         rtol=1e-8, atol=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(t_span[0], t_span[1], Nt)
    sol = solve_ivp(lambda t, W: vorticity_rhs(t, W, grid, cfg),
                    t_span, omega0.reshape(-1), t_eval=t_eval,
                    method="RK45", rtol=rtol, atol=atol)
    return sol.y.reshape(grid.N, grid.N, -1), t_eval


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(cfg: VorticityConfig,
                     T: float = 6.0, Nt: int = 600,
                     seeds=(1, 2, 3)) -> Dict[str, List]:
    """
    Generate a dataset of vorticity trajectories given a configuration.

    Args:
        cfg: VorticityConfig object (includes N, L, nu, U0, forcing).
        T: final time.
        Nt: number of timesteps.
        seeds: list of random seeds for different initial conditions.
    """
    grid = make_grid(cfg.N, cfg.L)
    omega_list, t_list, meta = [], [], []
    t_span = (0.0, T)

    for seed in tqdm(seeds, desc="Processing items"):
        rng = np.random.default_rng(seed)
        # Initial condition (can switch here for dipoles, random, etc.)
        omega0 = np.zeros((grid.N, grid.N))                # purely forcing-driven
        omega0 += ic_random_vorticity(grid,rng)
        omega_time, t_eval = integrate_trajectory(omega0, grid, cfg, t_span, Nt)

        omega_list.append(omega_time)
        t_list.append(t_eval)
        meta.append({"seed": seed,
                     "nu": cfg.nu,
                     "U0": cfg.U0,
                     "forcing": cfg.forcing})

    return {"omega_list": omega_list, "t_list": t_list,
            "grid": grid, "cfg": cfg, "meta": meta}

# ============================================================
# Noise model and visualization
# ============================================================

def add_noise(data, fixed_seed=0,
              sigma_add=0.10,      # additive std (in field units)
              sigma_mult=0.10,     # small gain jitter
              corr_len=0.006,       # spatial corr length in pixels (~interrogation size/2)
              ar1_phi=0.8,         # temporal lag-1 correlation (0.6–0.95 typical)
              alpha=1.0):          # gradient-based heteroscedasticity strength
    N, _, Nt = data.shape
    rng = np.random.default_rng(fixed_seed)

    # Build isotropic Gaussian spatial filter in Fourier domain
    k = 2*np.pi*np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    H = np.exp(-(KX**2 + KY**2) * (corr_len * N)**2 / 2.0)

    # Initialize AR(1) noise state
    eps = rng.standard_normal((N, N))
    
    # Precompute normalized gradient magnitude
    gx, gy = np.gradient(data, axis=(0,1))
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag /= grad_mag.max() + 1e-12  # normalize to [0,1]

    noisy = np.empty_like(data, dtype=np.float32)

    for t in range(Nt):
        if t > 0:
            # AR(1) with unit-variance innovations
            eps = ar1_phi * eps + np.sqrt(max(1e-12, 1 - ar1_phi**2)) * rng.standard_normal((N, N))
        e_spat = np.fft.ifft2(H * np.fft.fft2(eps)).real

        scale = 1.0 + alpha * grad_mag[:, :, t]
        mult = 1.0 + sigma_mult * rng.standard_normal((N, N))
        noisy[:, :, t] = (mult * data[:, :, t] + sigma_add * scale * e_spat).astype(np.float32)

    return noisy

def add_noise_deform(
    data,
    fixed_seed=0,
    sigma_add=0.3,          # additive noise std
    sigma_mult=0.2,         # multiplicative noise std
    corr_len=0.006,         # spatial correlation length
    ar1_phi=0.9,            # temporal correlation
    shear_factor=0.05,      # sinusoidal shear deformation
    deform_strength=0.02,   # smooth random deformation
    mult_clip=0.3,          # clip multiplicative factor
    preserve_rms=True       # rescale noisy frame to original RMS
):
    """
    Simplified noise + deformation for vorticity data.
    - Spatially & temporally correlated noise
    - Optional shear + smooth deformation
    - RMS preserved to avoid large amplitude blowup
    """
    N, _, Nt = data.shape
    noisy = np.zeros_like(data, dtype=np.float32)
    rng = np.random.default_rng(fixed_seed)

    # Fourier filter for spatial correlation
    k = 2*np.pi*np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    H = np.exp(-(KX**2 + KY**2) * (max(1, int(corr_len*N))**2) / 2.0)

    # AR(1) base noise state
    eps = rng.standard_normal((N, N))

    # Grids
    X, Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N), indexing="ij")
    smooth_filt = np.exp(-0.5*(KX**2 + KY**2)*(0.2**2))

    for t in range(Nt):
        # temporal correlation
        if t > 0:
            eps = ar1_phi*eps + np.sqrt(1 - ar1_phi**2) * rng.standard_normal((N, N))

        # spatial correlation
        e_spat = np.fft.ifft2(H * np.fft.fft2(eps)).real
        e_spat /= e_spat.std() + 1e-12

        # shear deformation of noise
        sx = shear_factor * rng.standard_normal()
        sy = shear_factor * rng.standard_normal()
        Xs = (X + sx*np.sin(2*np.pi*Y)) % 1.0
        Ys = (Y + sy*np.sin(2*np.pi*X)) % 1.0
        e_sheared = e_spat[(Xs*N).astype(int)%N, (Ys*N).astype(int)%N]

        # smooth random deformation of signal
        dx = deform_strength * rng.standard_normal((N,N))
        dy = deform_strength * rng.standard_normal((N,N))
        dx = np.fft.ifft2(np.fft.fft2(dx)*smooth_filt).real
        dy = np.fft.ifft2(np.fft.fft2(dy)*smooth_filt).real
        coords = np.array([((X+dx).ravel()*N) % N,
                           ((Y+dy).ravel()*N) % N])
        field_def = map_coordinates(data[:,:,t], coords, order=1, mode="wrap").reshape(N,N)

        # multiplicative jitter (bounded)
        mult = sigma_mult * rng.standard_normal((N,N))
        mult = np.clip(mult, -mult_clip, mult_clip)
        mult = 1.0 + mult

        # add noise
        noisy_frame = mult * field_def + sigma_add * e_sheared

        # preserve RMS if requested
        if preserve_rms:
            rms_clean = np.sqrt(np.mean(field_def**2))
            rms_noisy = np.sqrt(np.mean(noisy_frame**2))
            if rms_noisy > 1e-12:
                noisy_frame *= rms_clean / rms_noisy

        noisy[:,:,t] = noisy_frame.astype(np.float32)

    return noisy


def animate_fields(omega: np.ndarray, omega_noisy: np.ndarray,
                   interval_ms=30, filename="vorticity.mp4",
                   cmap="plasma", dpi=300, fps=30):
    """
    Animation of ω and ω_noisy on a black background.
    """
    Nt = omega.shape[2]
    clim = np.max(np.abs(omega_noisy))  # symmetric scaling

    # Create figure with black background
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi,
                             facecolor="black")
    ims = []

    for ax, f0, title in zip(
        axes,
        [omega[:, :, 0], omega_noisy[:, :, 0]],
        [r"$\omega$", r"$\omega_{\mathrm{noisy}}$"]
    ):
        im = ax.imshow(f0.T, origin="lower", cmap=cmap,
                       vmin=-clim, vmax=clim, animated=True)
        ax.set_title(title, fontsize=16, color="white")  # white labels
        ax.axis("off")
        ax.set_facecolor("black")
        ims.append(im)

    def update(i):
        ims[0].set_array(omega[:, :, i].T)
        ims[1].set_array(omega_noisy[:, :, i].T)
        return ims

    anim = FuncAnimation(fig, update, frames=range(0,Nt,10),
                         interval=interval_ms, blit=True)

    # Save high-quality animation
    if filename.endswith(".mp4"):
        anim.save(filename, dpi=dpi, fps=fps,
                  extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    elif filename.endswith(".gif"):
        anim.save(filename, dpi=dpi, fps=fps, writer="pillow")

    plt.close(fig)
    return anim
    
def add_noise_occlusion_bursts(
    data,
    fixed_seed=0,
    sigma_add=0.1,
    sigma_mult=0.05,
    corr_len=0.01,
    ar1_phi=0.9,
    n_bursts=3,
    burst_len=50,
    mask_radius=8,
    mask_amp=3.0
):
    """
    Noise with heteroskedastic schedule:
    - Moving occlusion: a disk mask with amplified noise travels across the field.
    - Bursts: global intervals with elevated noise.
    """
    N, _, Nt = data.shape
    rng = np.random.default_rng(fixed_seed)

    # Fourier filter for spatial correlation
    k = 2*np.pi*np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    H = np.exp(-(KX**2 + KY**2) * (corr_len * N)**2 / 2.0)

    eps = rng.standard_normal((N, N))
    noisy = np.empty_like(data, dtype=np.float32)

    # pre-sample burst start times
    starts = rng.choice(Nt-burst_len, size=n_bursts, replace=False)
    burst_mask = np.zeros(Nt, dtype=bool)
    for s in starts:
        burst_mask[s:s+burst_len] = True

    # occlusion trajectory
    cx, cy = rng.uniform(0.2*N, 0.8*N), rng.uniform(0.2*N, 0.8*N)
    vx, vy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)

    for t in range(Nt):
        if t > 0:
            eps = ar1_phi*eps + np.sqrt(max(1e-12, 1 - ar1_phi**2)) * rng.standard_normal((N, N))
        e_spat = np.fft.ifft2(H * np.fft.fft2(eps)).real
        e_spat /= e_spat.std() + 1e-12

        # multiplicative jitter
        mult = 1.0 + sigma_mult * rng.standard_normal((N, N))

        # base noise level
        s_add = sigma_add * (3.0 if burst_mask[t] else 1.0)

        # moving occlusion mask
        cx = (cx + vx) % N
        cy = (cy + vy) % N
        X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        mask = ((X-cx)**2 + (Y-cy)**2 <= mask_radius**2).astype(float)

        # inside mask: amplify noise
        noise_field = s_add * ((1-mask) * e_spat + mask_amp * mask * e_spat)

        noisy[:, :, t] = (mult * data[:, :, t] + noise_field).astype(np.float32)

    return noisy

if __name__ == "__main__":
    # Resolution and domain
    N = 128
    L = 2*np.pi
    dx = L / N
    nu = 1e-4

    # Background flow and cylinder parameters
    U0 = 1.5
    D  = 0.075*L          # ≈ 0.314
    St = 0.20

    # Broader, more separated lobes → larger vortices
    sigma = 0.38*D       # ≈ 0.12 (~8 Δx)
    offset = 0.50*D      # ≈ 0.16

    forcing = {
        "mode": "karman",
        "x_c": 0.15*L,
        "y_c": 0.50*L,
        "D":   D,
        "St":  St,
        "A":   13.0,         # tune 12–14
        "sigma": sigma,
        "offset": offset,
        "phi": 0.5*np.pi     # phase lag → staggered street
    }

    sponge = {
        "x0": 0.96,
        "width": 0.03,
        "rate": 400.0 * U0 / L
    }
    
    cfg = VorticityConfig(N=N, L=L, nu=nu, U0=U0, forcing=forcing, sponge=sponge)

    # Time integration long enough to form a street
    T  = 20.0
    Nt = 2000

    dataset = generate_dataset(cfg, T=T, Nt=Nt, seeds=range(1))  # zero IC → forcing-driven wake
    grid = dataset["grid"]
    
    for j in range(len(dataset["omega_list"])):
        # Access one trajectory to preview
        omega_clean = dataset["omega_list"][j]
        
        rng = np.random.default_rng(j)   # seed per run if you want reproducibility

        # Noise settings: shifted to be a bit stronger
        sigma_add=0.4,     # moderate additive noise
        sigma_mult=0.25,   # weaker multiplicative scaling
        corr_len=0.006,    # slightly longer correlation → smoother noise
        ar1_phi=0.9,       # keep strong temporal persistence
        amp_weight=1.5,    # still amplify strong vortices, but less
        grad_weight=0.5,   # same gradient weighting
        shear_factor=0.05, # lighter shear deformation
        gamma=1.5

        omega_noisy = add_noise(
            omega_clean, 
            fixed_seed=j,
            sigma_add=0.4,     # moderate additive noise
            sigma_mult=0.25,   # weaker multiplicative scaling
            corr_len=0.006,    # slightly longer correlation → smoother noise
            ar1_phi=0.9,       # keep strong temporal persistence
            amp_weight=1.5,    # still amplify strong vortices, but less
            grad_weight=0.5,   # same gradient weighting
            shear_factor=0.05, # lighter shear deformation
            gamma=1.5
        )
        
        if HAS_MPL:
            anim = animate_fields(
                omega_clean, omega_noisy,
                interval_ms=20,
                filename=f"./vorticity_street_{j}_nu_small.gif",
                cmap="RdBu_r",  # try plasma/inferno/turbo
                dpi=300,
                fps=(Nt/10)/T
            )

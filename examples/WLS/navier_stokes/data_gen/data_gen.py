import numpy as np
from tqdm import tqdm
from utils import (
    VorticityConfig,
    generate_dataset,
    add_noise,
    add_noise_deform,
    add_noise_deform_multiscale,
    make_grid
)

def get_noisy_dataset(
    N=128,
    L=2*np.pi,
    U0=1.5,
    Re=150.0,
    T=50.0,
    Nt=5000,
    seeds=range(15),
    dataset_path="vorticity_datasets_von_karman.npz",
    regenerate=True,
    sadd=0.2,
    smmult=0.2,
    corr=0.01,
    arr1=0.0,
    alpha=0.0,
    test=False
):
    """
    Returns clean and noisy vorticity datasets.

    Parameters
    ----------
    N : int
        Grid resolution (default 128).
    L : float
        Domain length (default 2π).
    U0 : float
        Reference velocity.
    Re : float
        Reynolds number (default 150).
    T : float
        Simulation time horizon.
    Nt : int
        Number of timesteps (default 5000).
    seeds : iterable
        Random seeds for multiple trajectories.
    dataset_path : str
        Path to saved dataset file.
    regenerate : bool
        If True, regenerate noisy data even if dataset file exists.

    Returns
    -------
    dataset_clean : np.ndarray
        Clean vorticity dataset, shape (num_sims, Nt, N, N).
    dataset_noisy : np.ndarray
        Noisy vorticity dataset, shape (num_sims, Nt, N, N).
    """

    if not regenerate:
        try:
            data = np.load(dataset_path)
            return data["clean"], data["noisy"]
        except FileNotFoundError:
            print(f"{dataset_path} not found. Generating new dataset...")

    # Simulation config
    D = 0.1 * L
    nu = U0 * D / Re
    forcing = {
        "mode": "bluff_penalized",
        "x_c": 0.15*L,
        "y_c": 0.50*L,
        "D":   D,
        "alpha":   300.0,
        "annulus": 0.03,
        "c_shell": 3.0,
        "seed_amp": 0.2,
    }
    sponge = {"x0": 0.98, "width": 0.01, "rate": 400.0*U0/L}
    cfg = VorticityConfig(N=N, L=L, nu=nu, U0=U0, forcing=forcing, sponge=sponge)

    # Grid for noise
    grid = make_grid(N, L)

    dataset_clean, dataset_noisy = [], []

    # If you want to use pre-generated clean data
    dataset_imp = np.load(dataset_path)['clean'].astype(np.float32)

    for j in tqdm(seeds, desc="Generating noisy dataset"):
        omega_clean = dataset_imp[j]
        if test:
            j=j+len(seeds)
        dataset_clean.append(omega_clean)

        # Add noise
        omega_noisy = add_noise( #add_noise_deform_multiscale(omega_clean,fixed_seed=j)
            omega_clean,
            fixed_seed=j,
            sigma_add=sadd,
            sigma_mult=smmult,
            corr_len=corr,
            ar1_phi=arr1,
            alpha=alpha,
        )
        dataset_noisy.append(omega_noisy)

    dataset_clean = np.array(dataset_clean, dtype=np.float32)
    dataset_noisy = np.array(dataset_noisy, dtype=np.float32)

    return dataset_clean, dataset_noisy


# Example usage in another script
if __name__ == "__main__":
    dataset_path="vorticity_datasets_von_karman.npz"
    clean, noisy = get_noisy_dataset()
    print("Clean shape:", clean.shape)
    print("Noisy shape:", noisy.shape)
    np.savez_compressed(dataset_path, clean=clean, noisy=noisy)
    

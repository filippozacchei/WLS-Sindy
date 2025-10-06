import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from data_gen import *
from var_encoder_def import *

# ============================================================
# Experiment grid
# ============================================================
noise_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
case = "noisy"

latent_dim = 4
batch_size, n_epochs = 32, 5

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)


# ============================================================
# Utility: normalize dataset with incremental stats
# ============================================================
def compute_stats(frames):
    # compute mean/std without materializing huge reshaped arrays
    s, ssq, count = 0.0, 0.0, 0
    for block in np.array_split(frames, max(1, len(frames) // 1000)):
        block = block.astype(np.float32)
        s += block.sum()
        ssq += (block**2).sum()
        count += block.size
    mean = s / count
    std = np.sqrt(ssq / count - mean**2)
    return float(mean), float(max(std, 1e-6))


def normalize(batch, mean, std):
    return ((batch - mean) / std).astype(np.float32)


# ============================================================
# Main loop over noise levels
# ============================================================
for smult in noise_levels:
    print(f"\n=== Training VAE for noise={smult:.3f} ===")

    # Load dataset for this noise case
    clean, noisy = get_noisy_dataset(
                sadd = smult,
                smmult = 0.00,
                corr = 0.01,
                arr1 = 0.9,
                alpha = 0.0, 
                regenerate=True, 
                seeds=range(5)
                )

    Ns, Nx, Ny, Nt = noisy.shape
    frames = noisy.transpose(0, 3, 1, 2).reshape(Ns * Nt, Nx, Ny)

    # Split train/test
    n_train = int(0.8 * len(frames))
    train_frames = frames[:n_train]
    test_frames = frames[n_train:]

    # Compute normalization statistics
    train_mean, train_std = compute_stats(train_frames)

    # Build datasets
    X_train = torch.from_numpy(normalize(train_frames, train_mean, train_std)[:, None, :, :])
    X_test = torch.from_numpy(normalize(test_frames, train_mean, train_std)[:, None, :, :])

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(X_test,  batch_size=batch_size, shuffle=False)

    # Model + optimizer
    model = ConvVAE(latent_dim, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    beta = 10
    # Training loop
    for epoch in range(n_epochs):
        kl_weight = min(1.0, epoch / 10)  # KL annealing

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Noise {smult:.3f} | Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                val_loss += vae_loss(recon, batch, mu, logvar, beta=beta).item() * len(batch)
        val_loss /= len(X_test)

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"train loss = {train_loss/len(X_train):.6e} | "
              f"val loss = {val_loss:.6e}")

    # Save model for this noise level
    out_path = f"vae_model_fc{latent_dim}_{case}_noise{smult:.3f}_beta{beta}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved -> {out_path}")

    # Free memory before next noise case
    del clean, noisy, frames, train_frames, test_frames, X_train, X_test, train_loader, val_loader, model


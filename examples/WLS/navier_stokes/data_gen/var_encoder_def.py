import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=256):
        super().__init__()
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 128 → 64
            nn.Conv2d(8, 16, 3, stride=2, padding=1),   # 64 → 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 → 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16 → 8
        )

        # Flatten (32×8×8 = 2048)
        self.fc_enc = nn.Sequential(
            nn.Linear(64*8*8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder projection
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64*8*8),
            nn.ReLU(),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 8 → 16
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16 → 32
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),   # 32 → 64
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),   # 64 → 128
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = self.fc_enc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)    
        return x_recon, mu, logvar

def vae_loss(recon, x, mu, logvar, beta=5):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta*kld) / x.size(0)


import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=256, dropout_p=0.2):
        super().__init__()
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 128 → 64
            nn.ReLU(),
            nn.Dropout2d(dropout_p),   # channel dropout
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32 → 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16 → 8
            nn.ReLU(),
        )

        # Flatten and fully connected layers (dense dropout here too)
        self.fc_enc = nn.Sequential(
            nn.Linear(64*8*8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Latent space
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder projection
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64*8*8),
            nn.GELU(),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),   # 64 → 128
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = self.fc_enc(h)
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# --- AE loss (simple MSE) ---
def ae_loss(recon, x):
    return nn.functional.mse_loss(recon, x, reduction='mean')

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------
def flat_dim(H=128, W=128, C=1):
    return C * H * W

def to_flat(x):   # (B,C,H,W) -> (B,D)
    return x.view(x.size(0), -1)

def to_img(y, H=128, W=128, C=1):  # (B,D) -> (B,C,H,W)
    return y.view(y.size(0), C, H, W)

# -------------------------
# MLP blocks
# -------------------------
class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0, act=nn.GELU):
        """
        dims: list like [in, h1, h2, ..., out]
        """
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:  # no norm/act on last layer
                layers += [nn.LayerNorm(dims[i+1]), act()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Deterministic Autoencoder (no CNN)
# -------------------------
class LinearAE(nn.Module):
    """
    Pure fully-connected AE for (B,1,128,128) inputs.
    """
    def __init__(self, H=128, W=128, C=1, latent_dim=32, width=1024, depth=3, dropout=0.0):
        super().__init__()
        D = flat_dim(H, W, C)

        # encoder
        enc_dims = [D] + [width]* (depth-1) + [latent_dim]
        self.encoder = MLP(enc_dims, dropout=dropout)

        # decoder (mirror)
        dec_dims = [latent_dim] + [width]* (depth-1) + [D]
        self.decoder = MLP(dec_dims, dropout=dropout)

        self.H, self.W, self.C = H, W, C

    def encode(self, x):
        return self.encoder(to_flat(x))

    def decode(self, z):
        y = self.decoder(z)
        return to_img(y, self.H, self.W, self.C)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

def ae_loss(x_hat, x, reduction='mean'):
    return F.mse_loss(x_hat, x, reduction=reduction)

# -------------------------
# Variational Autoencoder (same MLP backbone)
# -------------------------
class LinearVAE(nn.Module):
    """
    Fully-connected VAE (no CNN). Suitable when variance in the latent is required.
    """
    def __init__(self, H=128, W=128, C=1, latent_dim=32, width=1024, depth=3, dropout=0.0):
        super().__init__()
        D = flat_dim(H, W, C)

        # shared encoder trunk
        enc_dims = [D] + [width]* (depth-1)
        self.enc_trunk = MLP(enc_dims, dropout=dropout)
        self.fc_mu     = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-1], latent_dim)

        # decoder (mirror)
        dec_dims = [latent_dim] + [width]* (depth-1) + [D]
        self.decoder = MLP(dec_dims, dropout=dropout)

        self.H, self.W, self.C = H, W, C

    def encode(self, x):
        h = self.enc_trunk(to_flat(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        y = self.decoder(z)
        return to_img(y, self.H, self.W, self.C)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss_nn(x_hat, x, mu, logvar, beta=4.0, recon='mse'):
    if recon == 'mse':
        rec = F.mse_loss(x_hat, x, reduction='sum')
    elif recon == 'l1':
        rec = F.l1_loss(x_hat, x, reduction='sum')
    else:
        raise ValueError("recon must be 'mse' or 'l1'")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (rec + beta*kld) / x.size(0)

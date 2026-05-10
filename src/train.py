"""
DDPM Training Loop for the IMS Bearing Dataset.

Usage (CLI):
    python -m src.train --n_epochs 100 --batch_size 32 --lr 2e-4 \
                        --data_dir data/raw --channel 0 --timesteps 1000

Usage (programmatic, called from app.py):
    from src.train import train
    train(args)  # args has attributes: n_epochs, batch_size, lr, data_dir, channel, timesteps
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import IMSDataset
from src.diffusion import DDPMScheduler
from src.models import UNet1D


def train(args) -> None:
    """
    Train a UNet1D DDPM on healthy bearing data (first 500 files).

    Args:
        args: Namespace or object with attributes:
              n_epochs, batch_size, lr, data_dir, channel, timesteps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Dataset & DataLoader
    # ------------------------------------------------------------------ #
    dataset = IMSDataset(
        data_dir=args.data_dir,
        mode="train",
        channel=args.channel,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Train] Dataset: {len(dataset)} segments | Batches/epoch: {len(loader)}")

    # ------------------------------------------------------------------ #
    # 2. Model & Scheduler
    # ------------------------------------------------------------------ #
    timesteps = getattr(args, "timesteps", 1000)
    model = UNet1D(in_channels=1).to(device)
    scheduler = DDPMScheduler(T=timesteps).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] UNet1D parameters: {param_count:,}")

    # ------------------------------------------------------------------ #
    # 3. Training Loop
    # ------------------------------------------------------------------ #
    model.train()
    for epoch in range(1, args.n_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            x0 = batch.to(device)              # [B, 1, 4096]

            # Sample random timesteps uniformly from {1, ..., T}
            t = torch.randint(1, timesteps, (x0.shape[0],), device=device, dtype=torch.long)

            # Sample Gaussian noise
            eps = torch.randn_like(x0)

            # Forward diffusion: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * eps
            x_t = scheduler.add_noise(x0, eps, t)

            # Predict noise
            eps_hat = model(x_t, t)

            # MSE loss between true and predicted noise
            loss = loss_fn(eps_hat, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"[Train] Epoch [{epoch:>4}/{args.n_epochs}]  Loss: {avg_loss:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Save checkpoint
    # ------------------------------------------------------------------ #
    save_path = "unet_diffusion.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Train] Model saved → {save_path}")


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on IMS bearing data")
    parser.add_argument("--n_epochs",   type=int,   default=100,    help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,     help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,   help="Adam learning rate")
    parser.add_argument("--data_dir",   type=str,   default="data/raw", help="Path to raw IMS data")
    parser.add_argument("--channel",    type=int,   default=0,      help="Bearing channel index (0-indexed)")
    parser.add_argument("--timesteps",  type=int,   default=1000,   help="Number of diffusion timesteps T")
    args = parser.parse_args()
    train(args)

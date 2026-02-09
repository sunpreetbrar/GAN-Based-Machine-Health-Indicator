import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.models import Generator, Discriminator
from src.dataset import IMSDataset
from src.wgan import compute_gradient_penalty

def train(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Forcing CPU if user doesn't have CUDA setup optimized, but ideally CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found.")
        return

    dataset = IMSDataset(data_dir=args.data_dir, mode='train', segment_len=4096, channel=args.channel)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize models
    generator = Generator(latent_dim=args.latent_dim, signal_length=4096).to(device)
    discriminator = Discriminator(signal_length=4096).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Training Loop
    print(f"Starting training for {args.n_epochs} epochs...")
    
    for epoch in range(args.n_epochs):
        pbar = tqdm(dataloader)
        for i, real_imgs in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.shape[0]

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(batch_size, args.latent_dim, device=device)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)

            # Adversarial loss: D(fake) - D(real) + lambda_gp * gp
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                
                pbar.set_description(f"[Epoch {epoch}/{args.n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save model
    print("Saving discriminator...")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training") # Small default for quick test, user can increase
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=4096, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--lambda_gp", type=float, default=10, help="WGAN GP penalty lambda")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="path to dataset")
    parser.add_argument("--channel", type=int, default=0, help="channel index to use from dataset")
    
    args = parser.parse_args()
    train(args)

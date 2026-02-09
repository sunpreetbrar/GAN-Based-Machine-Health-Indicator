import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, signal_length=4096, channels=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        # We target 4096. We have 4 upsampling blocks of stride 4.
        # 4096 / 4 / 4 / 4 / 4 = 16.
        self.init_size = 16 
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            # 16 -> 64
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(inplace=True),

            # 64 -> 256
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(inplace=True),
            
            # 256 -> 1024
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),

            # 1024 -> 4096
            nn.ConvTranspose1d(32, channels, kernel_size=4, stride=4, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [Batch, 100]
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        signal = self.conv_blocks(out)
        return signal

class Discriminator(nn.Module):
    def __init__(self, signal_length=4096, channels=1):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=False):
            # Kernel 25, Stride 4, Padding 11 ensures output size is roughly input/4
            # (L+2P-K)/S + 1 = (L+22-25)/4 + 1 = (L-3)/4 + 1. 
            # If L=4096 -> 1023.25 + 1 -> 1024.
            block = [nn.Conv1d(in_filters, out_filters, 25, stride=4, padding=11)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            # 4096 -> 1024
            *discriminator_block(channels, 16, bn=False), 
            # 1024 -> 256
            *discriminator_block(16, 32, bn=False), 
            # 256 -> 64
            *discriminator_block(32, 64, bn=False),
            # 64 -> 16
            *discriminator_block(64, 128, bn=False),
            # Layer 5: 16 -> 4 
            *discriminator_block(128, 256, bn=False),
        )

        # 16 -> 4 (Flattened)
        # 256 channels * 4 length = 1024
        self.adv_layer = nn.Linear(256 * 4, 1)

    def forward(self, signal):
        out = self.model(signal)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

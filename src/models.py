import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Maps scalar timestep t → d-dimensional sinusoidal embedding."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] long timestep indices
        Returns:
            emb: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                                      # [half]
        args = t[:, None].float() * freq[None, :]             # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)     # [B, dim]
        return emb


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """
    GroupNorm → SiLU → Conv1d → GroupNorm → SiLU → Conv1d + residual skip.
    Time embedding injected via a linear projection added after the first norm.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 256, groups: int = 8):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        # Skip connection (1×1 conv if channels differ)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, in_ch, L]
            t_emb: [B, time_dim]
        Returns:
            out:   [B, out_ch, L]
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Inject time embedding
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]

        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# UNet1D
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    """
    1D U-Net noise predictor for DDPM on vibration signals.

    Architecture:
      Input  : [B, 1, 4096]
      Encoder stages (3):
        stage 0: ResBlock(64  → 128), Downsample(128) — L: 4096 → 2048
        stage 1: ResBlock(128 → 256), Downsample(256) — L: 2048 → 1024
        stage 2: ResBlock(256 → 512), Downsample(512) — L: 1024 → 512
      Bottleneck: ResBlock(512 → 512) at L=512
      Decoder stages (3):
        stage 0: Upsample(512→256), cat(skip=512) → 768, ResBlock(768→256) — L: 512 → 1024
        stage 1: Upsample(256→128), cat(skip=256) → 384, ResBlock(384→128) — L: 1024 → 2048
        stage 2: Upsample(128→64),  cat(skip=128) → 192, ResBlock(192→64)  — L: 2048 → 4096
      Output : GroupNorm → SiLU → Conv1d(64→1) → [B, 1, 4096]

    Skip connections carry the encoder ResBlock output BEFORE downsampling.
    """

    def __init__(self, in_channels: int = 1, time_dim: int = 256):
        super().__init__()

        # Channel progression: input_proj output, then encoder stages
        base_ch   = 64
        enc_chs   = [128, 256, 512]   # output channels per encoder stage
        self.enc_in_ch = [base_ch] + enc_chs[:-1]   # [64, 128, 256]

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Input projection: 1 → 64
        self.input_proj = nn.Conv1d(in_channels, base_ch, kernel_size=3, padding=1)

        # Encoder blocks (each: in_ch → out_ch) + stride-2 downsamplers
        self.enc_blocks = nn.ModuleList([
            ResBlock1D(in_c, out_c, time_dim)
            for in_c, out_c in zip(self.enc_in_ch, enc_chs)
        ])
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(out_c, out_c, kernel_size=4, stride=2, padding=1)
            for out_c in enc_chs
        ])

        # Bottleneck
        self.bottleneck = ResBlock1D(enc_chs[-1], enc_chs[-1], time_dim)

        # Decoder: upsample then concat with skip, then ResBlock
        # skip channels come from encoder outputs (reversed): [512, 256, 128]
        # upsampler input channels (reversed enc_chs):        [512, 256, 128]
        # upsampler output channels:                          [256, 128,  64]
        dec_out_chs = [256, 128, 64]
        skip_chs    = list(reversed(enc_chs))   # [512, 256, 128]
        up_in_chs   = list(reversed(enc_chs))   # [512, 256, 128]  (from prev dec stage out or bottleneck)
        # After up+concat: in_ch = dec_out_ch + skip_ch
        dec_in_chs  = [d + s for d, s in zip(dec_out_chs, skip_chs)]  # [768, 384, 192]

        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose1d(up_c, out_c, kernel_size=4, stride=2, padding=1)
            for up_c, out_c in zip(up_in_chs, dec_out_chs)
        ])
        self.dec_blocks = nn.ModuleList([
            ResBlock1D(in_c, out_c, time_dim)
            for in_c, out_c in zip(dec_in_chs, dec_out_chs)
        ])

        # Output projection: 64 → 1
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, dec_out_chs[-1]),
            nn.SiLU(),
            nn.Conv1d(dec_out_chs[-1], in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy signal  [B, 1, L]   (L = 4096)
            t: Timestep      [B] (long)
        Returns:
            Predicted noise  [B, 1, L]
        """
        t_emb = self.time_embed(t)            # [B, time_dim]

        # Input projection
        h = self.input_proj(x)                # [B, 64, L]

        # Encoder — save skip connections BEFORE downsampling
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamplers):
            h = enc(h, t_emb)                 # [B, out_ch, L]
            skips.append(h)                   # save before downsampling
            h = down(h)                       # [B, out_ch, L/2]

        # Bottleneck  [B, 512, L//8]
        h = self.bottleneck(h, t_emb)

        # Decoder — upsample, concat skip, ResBlock
        for up, dec, skip in zip(self.upsamplers, self.dec_blocks, reversed(skips)):
            h = up(h)
            # Handle potential size mismatch (odd-length signals)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)   # concat on channel axis
            h = dec(h, t_emb)

        return self.output_proj(h)            # [B, 1, L]

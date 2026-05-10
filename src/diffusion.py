import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """
    DDPM Noise Scheduler with linear beta schedule.

    Forward process: x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * eps
    Reverse process: single-step DDPM posterior mean denoising.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T

        # Linear beta schedule
        beta = torch.linspace(beta_start, beta_end, T)          # [T]
        alpha = 1.0 - beta                                        # [T]
        alpha_hat = torch.cumprod(alpha, dim=0)                  # [T]  ᾱ_t

        # Pre-compute commonly used quantities and register as buffers
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.register_buffer("sqrt_alpha_hat", alpha_hat.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_hat", (1.0 - alpha_hat).sqrt())
        self.register_buffer("sqrt_alpha", alpha.sqrt())
        self.register_buffer("sqrt_recip_alpha", alpha.rsqrt())

    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * eps

        Args:
            x0:  Clean signal  [B, 1, L]
            eps: Gaussian noise [B, 1, L]
            t:   Timestep indices (long) [B]

        Returns:
            x_t: Noisy signal [B, 1, L]
        """
        sqrt_ah = self.sqrt_alpha_hat[t].view(-1, 1, 1)          # [B, 1, 1]
        sqrt_one_minus_ah = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)
        return sqrt_ah * x0 + sqrt_one_minus_ah * eps

    @torch.no_grad()
    def denoise_step(self, model, x_t: torch.Tensor, t_scalar: int) -> torch.Tensor:
        """
        One reverse denoising step (DDPM posterior mean).

        Args:
            model:    Trained UNet1D noise predictor
            x_t:      Noisy signal at timestep t  [B, 1, L]
            t_scalar: Current timestep (Python int, same for entire batch)

        Returns:
            x_{t-1}: Less noisy signal [B, 1, L]
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_scalar, device=x_t.device, dtype=torch.long)

        eps_hat = model(x_t, t_tensor)                          # predicted noise

        beta_t = self.beta[t_scalar]
        alpha_t = self.alpha[t_scalar]
        alpha_hat_t = self.alpha_hat[t_scalar]
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t_scalar]
        sqrt_one_minus_ah_t = self.sqrt_one_minus_alpha_hat[t_scalar]

        # DDPM posterior mean: μ_θ(x_t, t)
        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_ah_t) * eps_hat)

        if t_scalar == 0:
            return mean
        else:
            # Add posterior variance noise: σ_t = sqrt(β_t)
            noise = torch.randn_like(x_t)
            sigma = beta_t.sqrt()
            return mean + sigma * noise

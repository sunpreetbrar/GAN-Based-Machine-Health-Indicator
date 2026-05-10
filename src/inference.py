"""
DDPM Health Indicator Inference for the IMS Bearing Dataset.

For each file (timestamp) in the full lifecycle:
  1. Normalize and segment the signal (same as training)
  2. Partial forward diffusion: jump to t_start (add controlled noise)
  3. Reverse denoising loop: run from t_start → 0 using the trained UNet
  4. Reconstruction Error: MSE(original, reconstructed) per segment, averaged per file
  5. Health Indicator: HI = exp(-MSE)
  6. Rolling-window smoothing (window=50)

Returns a DataFrame with columns: [Timestamp, MSE, Health_Indicator]
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.diffusion import DDPMScheduler
from src.models import UNet1D


def get_health_indicator(
    data_dir: str = "data/raw",
    model_path: str = "unet_diffusion.pth",
    channel: int = 0,
    t_start: int = 100,
    segment_len: int = 4096,
    smooth_window: int = 50,
) -> pd.DataFrame:
    """
    Compute a Health Indicator (HI) for every file in data_dir using DDPM reconstruction error.

    Args:
        data_dir:      Path to raw IMS bearing data files.
        model_path:    Path to the saved UNet1D checkpoint ('unet_diffusion.pth').
        channel:       Bearing channel index (0-indexed).
        t_start:       Partial noise level (10% of T=1000). Key hyperparameter.
        segment_len:   Length of each signal segment (default 4096).
        smooth_window: Rolling average window size for HI smoothing.

    Returns:
        DataFrame with columns [Timestamp, MSE, Health_Indicator], indexed 0…N-1.
        Returns an empty DataFrame if the model file is not found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    if not os.path.exists(model_path):
        print(f"[Inference] Model not found at '{model_path}'. Train first.")
        return pd.DataFrame()

    model = UNet1D(in_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scheduler = DDPMScheduler(T=1000).to(device)

    # ------------------------------------------------------------------ #
    # 2. Iterate over all files
    # ------------------------------------------------------------------ #
    data_path = Path(data_dir)
    files = sorted([f for f in data_path.iterdir() if f.is_file()])

    if not files:
        print(f"[Inference] No files found in '{data_dir}'.")
        return pd.DataFrame()

    records = []

    with torch.no_grad():
        for file_path in files:
            try:
                import pandas as _pd
                df_raw = _pd.read_csv(file_path, sep="\t", header=None)
                signal = df_raw.iloc[:, channel].values.astype(np.float32)
            except Exception as e:
                print(f"[Inference] Skipping {file_path.name}: {e}")
                continue

            # Normalize to [-1, 1] (per-file, matching training)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

            # Non-overlapping segmentation
            num_segments = len(signal) // segment_len
            if num_segments == 0:
                continue

            segments = [
                signal[i * segment_len : (i + 1) * segment_len]
                for i in range(num_segments)
            ]
            x0 = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(1).to(device)
            # x0: [num_segments, 1, segment_len]

            # ---- Partial forward diffusion: x_t_start ---- #
            eps = torch.randn_like(x0)
            t_tensor = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
            x_t = scheduler.add_noise(x0, eps, t_tensor)

            # ---- Reverse denoising loop: t_start → 0 ---- #
            for t_step in reversed(range(t_start + 1)):
                x_t = scheduler.denoise_step(model, x_t, t_step)

            x_reconstructed = x_t  # [num_segments, 1, segment_len]

            # ---- Reconstruction MSE (per file, averaged over segments) ---- #
            mse = float(
                ((x0 - x_reconstructed) ** 2).mean(dim=[1, 2]).mean().item()
            )

            # ---- Health Indicator ---- #
            hi = float(np.exp(-mse))

            records.append({
                "Timestamp": file_path.stem,
                "MSE": mse,
                "Health_Indicator": hi,
            })

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # 3. Smoothing
    # ------------------------------------------------------------------ #
    result_df["Health_Indicator"] = (
        result_df["Health_Indicator"]
        .rolling(window=smooth_window, min_periods=1, center=True)
        .mean()
    )

    return result_df

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from src.train import train
from src.inference import get_health_indicator
import numpy as np

# Streamlit App
st.set_page_config(page_title="DDPM Digital Twin", layout="wide")

st.title("Digital Twin Machine Health Monitor")
st.markdown("""
This dashboard visualizes the **Health Indicator (HI)** of a bearing over its lifecycle
using a **Denoising Diffusion Probabilistic Model (DDPM)** trained on healthy data.

**Method:** Reconstruction Error — healthy signals are reconstructed accurately (low MSE, HI ≈ 1.0),
while degraded signals produce high reconstruction error (HI → 0).
""")

# Sidebar
st.sidebar.header("Controls")

# Channel Selection
channel_index = st.sidebar.number_input(
    "Channel Index (0-indexed)", min_value=0, max_value=7, value=4,
    help="Set 1: Ch 4=Bearing 3. Set 2: Ch 0=Bearing 1."
)

# DDPM hyperparameters
class Args:
    n_epochs = 100
    batch_size = 32
    lr = 2e-4
    timesteps = 1000
    data_dir = "data/raw"
    channel = 0  # Overwritten below

args = Args()
args.channel = channel_index

# Advanced settings in sidebar expander
with st.sidebar.expander("Training Settings"):
    args.n_epochs = st.number_input("Epochs", min_value=1, max_value=500, value=100)
    args.batch_size = st.number_input("Batch Size", min_value=4, max_value=128, value=32)
    args.lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=2e-4, format="%.5f")

# Inference settings
t_start = st.sidebar.slider(
    "Noise Level (t_start)", min_value=10, max_value=500, value=100,
    help="Partial noise level for reconstruction. Higher = more noise added before denoising."
)

# Train Button
if st.sidebar.button("Train Model"):
    with st.spinner("Training DDPM on healthy data... This may take a while."):
        if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
            st.error(f"No data found in {args.data_dir}. Please add IMS dataset files.")
        else:
            try:
                train(args)
                st.success("Training Complete! Model saved as `unet_diffusion.pth`.")
            except Exception as e:
                st.error(f"An error occurred during training: {e}")

# Analyze Button
if st.sidebar.button("Analyze Lifecycle"):
    with st.spinner("Calculating Health Indicator for full lifecycle..."):
        if not os.path.exists("unet_diffusion.pth"):
            st.error("Model not found. Please train the model first.")
        else:
            df = get_health_indicator(data_dir=args.data_dir, model_path="unet_diffusion.pth", channel=channel_index, t_start=t_start)

            if not df.empty:
                st.success("Analysis Complete!")

                # Normalization Option
                normalize = st.checkbox(
                    "Normalize Score (0-1)", value=True,
                    help="Scale HI to [0, 1] range for paper comparison."
                )
                if normalize:
                    hi_min = df["Health_Indicator"].min()
                    hi_max = df["Health_Indicator"].max()
                    if hi_max > hi_min:
                        df["Health_Indicator"] = (df["Health_Indicator"] - hi_min) / (hi_max - hi_min)

                # Plot
                st.subheader("Bearing Health Degradation")

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df["Health_Indicator"], label="Health Indicator (Smoothed)", color="blue")

                try:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d.%H.%M.%S")
                    days = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds() / (24 * 3600)

                    tick_indices = np.linspace(0, len(df.index) - 1, 10).astype(int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([f"{int(round(days.iloc[i]))}" for i in tick_indices])
                    x_label = "Time (Days)"
                except Exception:
                    x_label = "Time (Files)"

                ax.axhline(y=0.5, color="red", linestyle="--", label="Warning Threshold")
                ax.set_xlabel(x_label)
                ax.set_ylabel("Health Indicator Score")
                ax.legend()
                st.pyplot(fig)

                # Show MSE plot
                st.subheader("Reconstruction Error (MSE)")
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(df.index, df["MSE"], color="orange", alpha=0.8)
                ax2.set_xlabel(x_label)
                ax2.set_ylabel("MSE")
                st.pyplot(fig2)

                # Show dataframe
                st.dataframe(df)
            else:
                st.warning("No results generated. Check data directory.")

st.sidebar.markdown("---")
st.sidebar.info("System: PyTorch DDPM UNet1D\nData: IMS Bearing Dataset")

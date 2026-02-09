import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from src.train import train
from src.inference import get_health_indicator
import argparse

# Streamlit App
st.set_page_config(page_title="Deep Digital Twin", layout="wide")

st.title("Digital Twin Machine Health Monitor")
st.markdown("""
This dashboard visualizes the **Health Indicator (HI)** of a bearing over its lifecycle using a **WGAN-GP** trained on healthy data.
""")

# Sidebar
st.sidebar.header("Controls")

# Channel Selection
channel_index = st.sidebar.number_input("Channel Index (0-indexed)", min_value=0, max_value=7, value=4, help="Set 1: Ch 4=Bearing 3. Set 2: Ch 0=Bearing 1.")

# Default args for training
class Args:
    n_epochs = 50
    batch_size = 32
    lr = 0.0001
    b1 = 0.5
    b2 = 0.9
    n_critic = 5
    latent_dim = 100
    img_size = 4096
    channels = 1
    sample_interval = 400
    lambda_gp = 10
    data_dir = "data/raw"
    channel = 0 # Default, will be overwritten

args = Args()
args.channel = channel_index

# Train Button
if st.sidebar.button("Train Model"):
    with st.spinner("Training WGAN-GP on healthy data... This may take a while."):
        # Check if data exists
        if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
            st.error(f"No data found in {args.data_dir}. Please add IMS dataset files.")
        else:
            try:
                # Run training
                # We need to capture stdout or just let it run.
                # For Streamlit, we might want to just run the function.
                # Adjust args if needed
                train(args)
                st.success("Training Complete! Model saved as `discriminator.pth`.")
            except Exception as e:
                st.error(f"An error occurred during training: {e}")

# Analyze Button
if st.sidebar.button("Analyze Lifecycle"):
    with st.spinner("Calculating Health Indicator for full lifecycle..."):
        if not os.path.exists("discriminator.pth"):
            st.error("Model not found. Please train the model first.")
        else:
            df = get_health_indicator(data_dir=args.data_dir, model_path="discriminator.pth", channel=channel_index)
            
            if not df.empty:
                st.success("Analysis Complete!")
                
                # Normalization Option
                normalize = st.checkbox("Normalize Score (0-1)", value=True, help="Scale HI to [0, 1] range for paper comparison.")
                if normalize:
                    df['Health_Indicator'] = (df['Health_Indicator'] - df['Health_Indicator'].min()) / (df['Health_Indicator'].max() - df['Health_Indicator'].min())

                
                # Convert timestamp if possible for better plotting
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d.%H.%M.%S')
                except:
                    pass # Keep as string or index if parsing fails

                # Plot
                st.subheader("Bearing Health Degradation")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df['Health_Indicator'], label="Health Indicator (Smoothed)", color='blue')
                
                # Threshold line (Example: 0.0)
                # Note: HI from WGAN discriminator is unbounded logits usually, but often centered around 0 or positive/negative.
                # WGAN-GP discriminator tries to maximize D(real) - D(fake).
                # Healthy samples (real) should have high scores.
                # Anomalous samples (degraded) should validly have lower scores (closer to fake or just different).
                # User prompted: "If HI drops below threshold (e.g., 0.0), mark as 'Failure Detected'."
                threshold = 0.0
                ax.axhline(y=threshold, color='red', linestyle='--', label="Failure Threshold")
                
                ax.set_xlabel("Time (Files)")
                ax.set_ylabel("Health Indicator Score")
                ax.legend()
                st.pyplot(fig)
                
                # Show dataframe
                st.dataframe(df)
            else:
                st.warning("No results generated. Check data directory.")

st.sidebar.markdown("---")
st.sidebar.info("System: PyTorch WGAN-GP\nData: IMS Bearing Dataset")

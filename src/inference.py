import torch
import pandas as pd
import numpy as np
from src.models import Discriminator
from src.dataset import IMSDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_health_indicator(data_dir="data/raw", model_path="discriminator.pth", channel=0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path}...")
    try:
        model = Discriminator(signal_length=4096)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return pd.DataFrame()

    # Use 'test' mode to get all files
    dataset = IMSDataset(data_dir=data_dir, mode='test', segment_len=4096)
    
    # We want to process file by file to average scores per file (timestamp)
    # But Dataset flattens everything into segments.
    # To map segments back to files, we can rely on order if dataset preserves it (it does).
    # Alternatively, we can manually iterate files as in the prompt logic.
    # "Iterate through the entire dataset (all timestamps) sequentially."
    
    results = []
    
    # Iterate manually through files to keep timestamp structure
    print("Calculating Health Indicator...")
    files = dataset.files 
    
    with torch.no_grad():
        for i, file_path in enumerate(tqdm(files)):
            try:
                # Load and process single file matches dataset logic
                df = pd.read_csv(file_path, sep='\t', header=None)
                signal = df.iloc[:, channel].values.astype(np.float32)
                
                # Normalize (same as training)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
                signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
                
                # Segment
                segments = []
                num_segments = len(signal) // 4096
                for j in range(num_segments):
                    segments.append(signal[j*4096 : (j+1)*4096])
                
                if not segments:
                    continue
                    
                batch = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(1).to(device)
                
                # Inference
                outputs = model(batch)
                
                # Mean score for this file (timestamp)
                hi_score = outputs.mean().item()
                
                # Timestamp approximation: index or parse filename if it's a timestamp
                # IMS filenames are like "2003.10.22.12.06.24" -> %Y.%m.%d.%H.%M.%S
                filename = file_path.name
                results.append({"Timestamp": filename, "Health_Indicator": hi_score})
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df_res = pd.DataFrame(results)
    
    # Smoothing
    if not df_res.empty:
        df_res["Health_Indicator"] = df_res["Health_Indicator"].rolling(window=50, min_periods=1).mean()
        
    return df_res

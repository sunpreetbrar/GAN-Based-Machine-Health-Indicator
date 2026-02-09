import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class IMSDataset(Dataset):
    def __init__(self, data_dir, mode='train', segment_len=4096, channel=0):
        """
        Args:
            data_dir (str): Path to data directory (e.g., 'data/raw').
            mode (str): 'train' (first 500 files) or 'test' (all files).
            segment_len (int): Length of each segment (default: 4096).
            channel (int): Column index to extract (default: 0 for Bearing 3/4 based on user input, usually 0 is fine). 
                           Note: IMS dataset structure varies. Assuming 1st column is the target.
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.segment_len = segment_len
        self.channel = channel
        
        # 1. Loader: Sort files by timestamp
        self.files = sorted([f for f in self.data_dir.iterdir() if f.is_file()])
        
        # 2. Split: Train (first 500), Test (all/remaining)
        # "Train only on the 'healthy' data (first 20% of files)" - user prompt
        # User prompt also said: "train: Return segments from the first 500 files"
        if mode == 'train':
            self.files = self.files[:500]
        # test mode uses all files per instructions
        
        self.segments = []
        self._load_and_segment()

    def _load_and_segment(self):
        """
        Loads files, normalizes, and segments the data.
        """
        print(f"Loading {len(self.files)} files for {self.mode} mode...")
        
        for file_path in self.files:
            try:
                # IMS data is tab separated, no header
                df = pd.read_csv(file_path, sep='\t', header=None)
                signal = df.iloc[:, self.channel].values.astype(np.float32)
                
                # 4. Normalization: Normalize to [-1, 1]
                # Note: MinMax should ideally be fitted on training set global stats, 
                # but per-sample or per-file normalization is common in some GAN setups.
                # However, for a digital twin, global normalization is safer.
                # Here, following "Normalize signals to range [-1,1]" instruction literally per file for simplicity 
                # or we can do it on the segment. Let's do it per file to maintain relative amplitude within file.
                scaler = MinMaxScaler(feature_range=(-1, 1))
                signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

                # 2. Segmentation
                # "Implementing a sliding window or random crop to split the 20,480-point signal into segments of length 4096."
                # We will use non-overlapping sliding window for simplicity and coverage.
                num_segments = len(signal) // self.segment_len
                for i in range(num_segments):
                    start = i * self.segment_len
                    end = start + self.segment_len
                    segment = signal[start:end]
                    self.segments.append(segment)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # Return [1, 4096] tensor
        segment = self.segments[idx]
        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

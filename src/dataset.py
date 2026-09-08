import torch
import numpy as np
from torch.utils.data import Dataset

class MSIDataset(Dataset):
    def __init__(self, A_list, B_list):
        self.A_list = A_list
        self.B_list = B_list
        
    def __len__(self):
        return len(self.B_list)
    
    def __getitem__(self, idx):
        # 1. B (Observation) -> [M, H, W]
        B = self.B_list[idx] 
        # The last raw B dimension is the measured m/z dimension M.  Use it to
        # orient A deterministically; isotope expansion can make L > M, so the
        # previous size-based heuristic ("M is usually larger than L") is not
        # valid for the Adapter library.
        observed_mz_count = B.shape[-1]
        # 处理可能的维度问题
        if B.ndim == 3:
            B = torch.as_tensor(B).float().permute(2, 0, 1) # [H,W,M] -> [M,H,W]
        elif B.ndim == 4: # 如果数据是 [1, H, W, M]
            B = torch.as_tensor(B).float().squeeze(0).permute(2, 0, 1)
            
        # 2. A (Library) -> [M, L]
        A = torch.as_tensor(self.A_list[idx]).float()
        
        # 自动处理转置问题，确保是 [M, L]
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D [M,L] or [L,M], got {tuple(A.shape)}")
        if A.shape[0] != observed_mz_count and A.shape[1] == observed_mz_count:
            A = A.T
        if A.shape[0] != observed_mz_count:
            raise ValueError(
                f"A/B m/z mismatch: A={tuple(A.shape)}, B raw={self.B_list[idx].shape}"
            )
            
        return A, B 

def load_data(A_path, B_path):
    """加载原始 .npy 数据"""
    try:
        A = np.load(A_path, allow_pickle=True)
        B = np.load(B_path, allow_pickle=True)
        return A, B
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

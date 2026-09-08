import os
import torch
import random
import numpy as np
import shutil

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_A_matrix(path, device):
    """加载并标准化 A 矩阵"""
    try:
        A = np.load(path)
        if A.ndim == 3:
            # Processed libraries use the explicit [1, M, L] contract.
            A = A[0]
        elif A.ndim != 2:
            raise ValueError(f"A must be [1,M,L] or [M,L], got {A.shape}")

        # Do not infer orientation from relative axis sizes.  Isotope expansion
        # can legitimately produce more candidates (L) than m/z bins (M).
            
        A_tensor = torch.as_tensor(A).float().to(device)
        
        # 列归一化 (L2 Norm)
        # 这一步非常重要，保证了不同数据的 A 都在同一个量级
        col_norms = torch.norm(A_tensor, p=2, dim=0, keepdim=True)
        A_tensor = A_tensor / (col_norms + 1e-8)
        
        return A_tensor
    except Exception as e:
        print(f"❌ Error loading A: {e}")
        return None

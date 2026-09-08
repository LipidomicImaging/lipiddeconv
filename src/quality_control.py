import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage import measure, filters

def compute_tissue_mask(X_total, percentile=20):
    """
    自动检测组织区域
    X_total: [H, W] 所有脂质叠加图
    """
    # Otsu 自动阈值分割
    thresh = filters.threshold_otsu(X_total.cpu().numpy() if torch.is_tensor(X_total) else X_total)
    mask = (X_total > thresh) if torch.is_tensor(X_total) else (X_total > thresh)
    return mask

def compute_aggregation_index(X_single):
    """
    计算聚集度指数（Aggregation Index）
    高值 = 信号集中在少数热点（真脂质）
    低值 = 信号弥散（假阳性/背景）
    """
    # Gini coefficient 的简化版
    x_sorted = torch.sort(X_single.flatten())[0]
    n = x_sorted.shape[0]
    cumsum = torch.cumsum(x_sorted, dim=0)
    gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
    return gini  # 0=完全均匀，1=完全集中在一点

def compute_lipid_quality_metrics_v2(X, B_obs, A_curr, uncertainty_map, 
                                     meta_data=None,  # 新增：需要碎片信息
                                     bg_percentile=10):
    """
    版本 2：基于生物物理约束的 QC
    """
    B, L, H, W = X.shape
    device = X.device
    
    # 1. 组织掩码约束（硬约束）
    X_total = X.sum(dim=1).squeeze(0)  # [H, W]
    tissue_mask = compute_tissue_mask(X_total)  # [H, W]
    tissue_mask = tissue_mask.to(device)
    
    tissue_area = tissue_mask.sum().item()
    metrics = {}
    
    # 2. 组织内信号占比（In-Tissue Ratio）
    in_tissue_ratios = []
    for l in range(L):
        lipid_map = X[0, l]  # [H, W]
        total_sig = lipid_map.sum()
        tissue_sig = (lipid_map * tissue_mask).sum()
        ratio = tissue_sig / (total_sig + 1e-8)
        in_tissue_ratios.append(ratio.item())
    metrics['in_tissue_ratio'] = torch.tensor(in_tissue_ratios, device=device)
    
    # 3. 聚集度指数（Aggregation Index）- 替代 Spatial Coherence
    agg_indices = []
    for l in range(L):
        # 只在组织内计算聚集度（排除背景干扰）
        masked_map = X[0, l] * tissue_mask
        if masked_map.sum() > 0:
            agg = compute_aggregation_index(masked_map)
        else:
            agg = torch.tensor(0.0, device=device)
        agg_indices.append(agg.item())
    metrics['aggregation_index'] = torch.tensor(agg_indices, device=device)
    
    # 4. 谱图物理一致性（Fragment Consistency）
    # 检查该 lipid 是否使用了合理的碎片组合
    # 简化版：计算 A 矩阵中非零元素的离散度（真脂质有特定碎片模式）
    A_np = A_curr.cpu().numpy()  # [M, L]
    frag_sparsity = (np.abs(A_np) > 1e-6).sum(axis=0) / A_np.shape[0]  # [L]
    # 真脂质的碎片模式通常是稀疏的（特定几个 m/z），假阳性可能用到很多碎片
    metrics['fragment_sparsity'] = torch.tensor(frag_sparsity, device=device)
    
    # 5. 重建 fidelity（与观测的局部相关性）
    B_rec = torch.einsum('ml,blhw->bmhw', A_curr, X)
    local_corrs = []
    for l in range(L):
        # 计算该 lipid 的空间分布与重建残差的局部相关性
        X_l = X[0, l]  # [H, W]
        B_obs_local = B_obs[0, :, :, :].mean(dim=0)  # 平均谱图 [H, W]
        B_rec_local = B_rec[0, :, :, :].mean(dim=0)
        
        # 只在组织内计算
        mask_flat = tissue_mask.flatten()
        x_flat = X_l.flatten()[mask_flat]
        obs_flat = B_obs_local.flatten()[mask_flat]
        
        if x_flat.std() > 0 and obs_flat.std() > 0:
            corr = F.cosine_similarity(x_flat.unsqueeze(0), obs_flat.unsqueeze(0))
            local_corrs.append(corr.item())
        else:
            local_corrs.append(0.0)
    metrics['local_correlation'] = torch.tensor(local_corrs, device=device)
    
    # 6. 总强度（用于弱信号过滤）
    metrics['total_intensity'] = X.sum(dim=[0, 2, 3])
    
    return metrics, tissue_mask

def filter_false_positives_v2(X, metrics, 
                              in_tissue_threshold=0.75,  # 组织内信号必须 >75%
                              aggregation_threshold=0.4,  # 聚集度必须 >0.4（不能太均匀）
                              min_intensity_percentile=30):
    """
    基于硬约束的过滤（非加权平均）
    """
    L = X.shape[1]
    
    # 硬约束 1：必须在组织内
    mask1 = metrics['in_tissue_ratio'] > in_tissue_threshold
    
    # 硬约束 2：必须聚集（不能是弥散噪声）
    mask2 = metrics['aggregation_index'] > aggregation_threshold
    
    # 硬约束 3：不能是"碎片饥渴"（用了太多碎片的通常是假阳性）
    # 真脂质通常只用 3-5 个特征碎片，假阳性可能为了拟合数据动用 20+ 个碎片
    mask3 = metrics['fragment_sparsity'] < 0.3  # 使用的碎片占比 < 30%
    
    # 硬约束 4：局部相关性不能为负（与观测反相关的肯定是错的）
    mask4 = metrics['local_correlation'] > 0.1
    
    # 软约束 5：总强度不能太低
    min_int = torch.quantile(metrics['total_intensity'], min_intensity_percentile/100)
    mask5 = metrics['total_intensity'] > min_int
    
    # 综合（必须满足所有硬约束）
    valid_mask = mask1 & mask2 & mask3 & mask4 & mask5
    
    # 计算存在性分数（仅用于排序，不用于硬过滤）
    # 使用更稳健的加权（避免之前那种中庸分数）
    score = (metrics['in_tissue_ratio'] * 0.4 + 
             metrics['aggregation_index'] * 0.3 + 
             metrics['local_correlation'] * 0.3)
    
    # 对于被过滤掉的，分数设为 0
    score = score * valid_mask.float()
    
    kept_indices = torch.where(valid_mask)[0].cpu().numpy()
    X_filtered = X[:, valid_mask, :, :]
    
    return valid_mask, X_filtered, kept_indices, score
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridUnmixingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def tv_loss(self, x):
        """
        Total Variation Loss (各向异性)
        作用：鼓励空间上的连续性。
        对于同分异构体，即使强度低，只要是成片存在的，TV Loss 就会保护它。
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = x[:, :, 1:, :].numel()
        count_w = x[:, :, :, 1:].numel()
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def forward(self, X_layers, B_obs, A_curr, A_init):
        loss_total = 0.0
        info = {}
        K = len(X_layers)
        
        # 深监督：越深层权重越大
        layer_weights = [i/K for i in range(1, K+1)]
        
        for k, X in enumerate(X_layers):
            w = layer_weights[k]
            
            # --- 1. Reconstruction Loss (L1) ---
            # 物理保真：A*X 必须等于 B
            B_hat = torch.einsum('ml, blhw -> bmhw', A_curr, X)
            l_rec = F.l1_loss(B_hat, B_obs)
            
            # --- 2. Cosine Loss ---
            # 形状匹配：忽略绝对强度，只看指纹
            cos_sim = F.cosine_similarity(B_hat, B_obs, dim=1).mean()
            l_cos = 1.0 - cos_sim
            
            # --- 3. Weak Sparsity (L1) ---
            # 仅用于抑制纯背景噪音，权重极低
            l_spa = torch.mean(torch.abs(X))
            
            # --- 4. Spatial Continuity (TV) ---
            # 保护异构体结构
            l_tv = self.tv_loss(X)
            
            # 组合
            layer_loss = (self.cfg.lambda_recon * l_rec +
                          self.cfg.lambda_cosine * l_cos +
                          self.cfg.lambda_sparse * l_spa +
                          self.cfg.lambda_tv * l_tv)
            
            loss_total += w * layer_loss
            
            # 记录最后一层的数据
            if k == K-1:
                info['Rec'] = l_rec.item()
                info['Cos'] = l_cos.item()
                info['TV']  = l_tv.item()
        
        # --- 5. Anchor Loss ---
        # 防止 A 漂移：惩罚 A_curr 与 A_init 的偏差
        l_anchor = F.mse_loss(A_curr, A_init)
        loss_total += self.cfg.lambda_anchor * l_anchor
        info['Anchor'] = l_anchor.item()
        
        return loss_total, info
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    """
    基于同方差不确定性 (Homoscedastic Uncertainty) 的自动多任务损失加权。
    参考文献: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018.
    
    原理: Loss = (1 / 2*sigma^2) * L_task + log(sigma)
    sigma 越大，说明该任务越难/数值越大，权重就会自动变小。
    """
    def __init__(self, num_losses=5):
        super().__init__()
        # 初始化 log_variance (s) 为 0，即 sigma = 1
        # 使用 Parameter 使得它可以被优化器学习
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_list):
        """
        输入: 一个包含标量 Loss 的列表 [L_recon, L_cos, L_spa, L_tv, L_anc]
        输出: 加权后的总 Loss
        """
        total_loss = 0
        
        for i, loss in enumerate(loss_list):
            # 获取第 i 个任务的不确定性参数 s
            s = self.params[i]
            s = torch.clamp(s, min=-9.0, max=3.0) 
            
            # 计算加权 Loss
            # precision = exp(-s)
            precision = 0.5 * torch.exp(-s)
            
            # Weighted Loss + Regularizer (log(sigma))
            # 注意: log(sigma) = log(sqrt(exp(s))) = 0.5 * s
            total_loss += precision * loss + 0.5 * s
            
        return total_loss

    def get_current_weights(self):
        """返回当前的实际权重 (exp(-s))，用于监控"""
        with torch.no_grad():
            weights = 0.5 * torch.exp(-self.params)
        return weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 独立多尺度注意力模块 (Channel-Independent Multi-Scale Attention)
# 真正做到每个脂质通道独立提取 1x1, 3x3, 5x5 空间特征阿，并自适应融合
# ==========================================
class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # --- 通道注意力 (Channel Attention) ---
        # 这一步保留，用于评估每种脂质在当前迭代中的整体丰度重要性
        mid_planes = max(channels // reduction, 4)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, mid_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_planes, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # --- 独立多尺度空间注意力 (核心重构) ---
        # 关键点：使用 groups=channels，让每个通道独立进行卷积，绝对不混合！
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        
        # 自适应融合 (Merge & Attention)
        # 输入是 3*channels，输出是 channels，按照 groups=channels 独立融合
        # 意味着：对于第 i 个脂质，它专属的 1x1, 3x3, 5x5 特征会被融合为一个专属的 Mask
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, groups=channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- 1. 通道级特征校准 ---
        avg_out = self.channel_fc(F.adaptive_avg_pool2d(x, 1))
        max_out = self.channel_fc(F.adaptive_max_pool2d(x, 1))
        x = x * (avg_out + max_out)

        # --- 2. 逐通道独立多尺度空间特征提取 ---
        # 不做任何维度压缩！直接在原特征图上提取 [B, L, H, W]
        out1 = F.relu(self.branch1(x))
        out2 = F.relu(self.branch2(x))
        out3 = F.relu(self.branch3(x))
        
        # --- 3. 维度重排与融合 ---
        # 为了让融合卷积 (groups=channels) 正确识别同一通道的 3 个尺度，
        # 我们需要在维度 2 上进行 Stack，然后 Reshape，
        # 这样张量的通道顺序就会变成：[Lipid1_s1, Lipid1_s2, Lipid1_s3, Lipid2_s1, Lipid2_s2...]
        merged = torch.stack([out1, out2, out3], dim=2) 
        merged = merged.view(x.shape[0], x.shape[1] * 3, x.shape[2], x.shape[3]) # [B, L*3, H, W]
        
        # 为每一个脂质通道独立生成 0~1 的空间掩码，输出形状 [B, L, H, W]
        ms = self.fusion(merged) 
        
        # 应用掩码
        x = x * ms
        
        return x, ms

# ==========================================
# 2. 弹性阈值模块 (集成多尺度注意力)
# ==========================================
class ElasticThreshold(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mc_dropout = nn.Dropout2d(p=0.1)
        self.lambda1 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.000001)
        self.lambda2 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.000001)
        
        self.attention = MultiScaleAttentionBlock(channels)

    def forward(self, x):
        x_att, mask = self.attention(x)
        x_att = self.mc_dropout(x_att)
        
        # 当 mask 接近 1 (高置信区域) 时，阈值几乎降为 0
        l1 = F.relu(self.lambda1 * (1.0 + 1e-4 - mask)) 
        l2 = F.relu(self.lambda2)
        
        x_soft = F.relu(x - l1) 
        factor = 1.0 / (1.0 + l2)
        
        return x_soft * factor, mask

# ==========================================
# 3. ISTA Block
# ==========================================
class ElasticISTABlock(nn.Module):
    def __init__(self, M, L):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(0.1))
        self.thresholder = ElasticThreshold(L)
        self.momentum = nn.Parameter(torch.tensor(0.8))

    def forward(self, X_k, X_prev, B_obs, A):
        B_rec = torch.einsum('ml, blhw -> bmhw', A, X_k)
        Residual = B_rec - B_obs
        Grad = torch.einsum('ml, bmhw -> blhw', A, Residual)
        
        Z_k = X_k - self.step_size * Grad
        
        if X_prev is not None:
            Z_k = Z_k + self.momentum * (X_k - X_prev)
            
        X_next, mask = self.thresholder(Z_k) 
        
        return X_next, mask

# ==========================================
# 4. 光谱校准
# ==========================================
class SpectralCalibration(nn.Module):
    def __init__(self, M, L, A_init, clamp_min=0.8, clamp_max=1.2):
        super().__init__()
        self.register_buffer('A_base', A_init)
        self.W = nn.Parameter(torch.ones(M, L))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def get_calibrated_A(self):
        w_clamped = torch.clamp(self.W, self.clamp_min, self.clamp_max)
        return self.A_base * w_clamped

# ==========================================
# 5. 主网络 (LipidENNet)
# ==========================================
class LipidENNet(nn.Module):
    def __init__(self, A_init, K=10, clamp_min=0.8, clamp_max=1.2):
        super().__init__()
        M, L = A_init.shape
        self.calibrator = SpectralCalibration(M, L, A_init, clamp_min, clamp_max)
        self.layers = nn.ModuleList([ElasticISTABlock(M, L) for _ in range(K)])
        self.init_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, B_obs):
        A_curr = self.calibrator.get_calibrated_A()
        
        X = torch.einsum('ml, bmhw -> blhw', A_curr, B_obs) * self.init_scale
        X = F.relu(X)
        
        X_prev = None
        X_layers = []
        masks = []
        
        for layer in self.layers:
            X_next, mask = layer(X, X_prev, B_obs, A_curr)
            X_prev = X
            X = X_next
            X_layers.append(X)
            masks.append(mask)
            
        return X, X_layers, A_curr, masks


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ==========================================
# # 1. 独立多尺度注意力模块 (保留了完整的双通道注意力)
# # ==========================================
# class MultiScaleAttentionBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
        
#         # --- 通道注意力 (Channel Attention) ---
#         # 评估每种脂质在当前迭代中的整体丰度重要性
#         mid_planes = max(channels // reduction, 4)
#         self.channel_fc = nn.Sequential(
#             nn.Conv2d(channels, mid_planes, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(mid_planes, channels, 1, bias=False),
#             nn.Sigmoid()
#         )
        
#         # --- 独立多尺度空间注意力 (Spatial Attention) ---
#         # 核心：groups=channels 让每个通道独立进行卷积，绝对不混合！
#         self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False)
#         self.branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
#         self.branch3 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        
#         # 自适应融合：将独立提取的 1x1, 3x3, 5x5 特征融合为专属掩码
#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels * 3, channels, kernel_size=1, groups=channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # --- 1. 通道级特征校准 ---
#         avg_out = self.channel_fc(F.adaptive_avg_pool2d(x, 1))
#         max_out = self.channel_fc(F.adaptive_max_pool2d(x, 1))
#         x = x * (avg_out + max_out)

#         # --- 2. 逐通道独立多尺度空间特征提取 ---
#         out1 = F.relu(self.branch1(x))
#         out2 = F.relu(self.branch2(x))
#         out3 = F.relu(self.branch3(x))
        
#         # --- 3. 维度重排与融合 ---
#         merged = torch.stack([out1, out2, out3], dim=2) 
#         merged = merged.view(x.shape[0], x.shape[1] * 3, x.shape[2], x.shape[3]) # [B, L*3, H, W]
        
#         # 为每一个脂质通道独立生成 0~1 的空间掩码 [B, L, H, W]
#         ms = self.fusion(merged) 
        
#         # 应用掩码获取注意力加权后的特征
#         x_att = x * ms
        
#         return x_att, ms

# # ==========================================
# # 2. 弹性阈值模块 (集成多尺度注意力 & 修复通道截断)
# # ==========================================
# class ElasticThreshold(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.mc_dropout = nn.Dropout2d(p=0.1)
        
#         # 【修改 1】提高初始空间阈值 (从 1e-6 提升到 0.05)
#         # 激活 ISTA 网络的物理截断能力，起步就压制背景散粒噪声
#         self.lambda1 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.005)
#         self.lambda2 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.001)
        
#         # 【修改 2】新增：全局通道阈值 (Group Sparsity 的核心)
#         # 用来一刀切地砍掉竞争失败的假阳性脂质通道（例如幽灵般的 PE O-16:2）
#         self.channel_threshold = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.08)
        
#         self.attention = MultiScaleAttentionBlock(channels)

#     def forward(self, x):
#         # 1. 获取双注意力加权后的特征和空间掩码
#         x_att, mask = self.attention(x)
        
#         # 2. 空间级软阈值 (Spatial Soft-thresholding)
#         # 结合掩码：高置信区域阈值极小，背景区域阈值变大
#         l1 = F.relu(self.lambda1 * (1.0 + 1e-4 - mask)) 
        
#         # 【修复 Bug】作用于注意力加权后的 x_att，并利用 ReLU 保证丰度非负
#         x_spatial_soft = F.relu(x_att - l1) 
        
#         # 3. 通道级软阈值 (Channel Soft-thresholding / Group Sparsity)
#         # 如果该通道在空间上的残存信号连基础底线 t_c 都达不到，整个通道置为 0
#         t_c = F.relu(self.channel_threshold)
#         x_channel_soft = F.relu(x_spatial_soft - t_c)
        
#         # 4. 对真正确信的特征进行 Dropout 不确定性估计
#         x_dropped = self.mc_dropout(x_channel_soft)
        
#         # 弹性网络的 L2 缩放
#         l2 = F.relu(self.lambda2)
#         factor = 1.0 / (1.0 + l2)
        
#         return x_dropped * factor, mask

# # ==========================================
# # 3. ISTA Block
# # ==========================================
# class ElasticISTABlock(nn.Module):
#     def __init__(self, M, L):
#         super().__init__()
#         self.step_size = nn.Parameter(torch.tensor(0.1))
#         self.thresholder = ElasticThreshold(L)
#         self.momentum = nn.Parameter(torch.tensor(0.8))

#     def forward(self, X_k, X_prev, B_obs, A):
#         B_rec = torch.einsum('ml, blhw -> bmhw', A, X_k)
#         Residual = B_rec - B_obs
#         Grad = torch.einsum('ml, bmhw -> blhw', A, Residual)
        
#         Z_k = X_k - self.step_size * Grad
        
#         if X_prev is not None:
#             Z_k = Z_k + self.momentum * (X_k - X_prev)
            
#         X_next, mask = self.thresholder(Z_k) 
        
#         return X_next, mask

# # ==========================================
# # 4. 光谱校准
# # ==========================================
# class SpectralCalibration(nn.Module):
#     def __init__(self, M, L, A_init, clamp_min=0.8, clamp_max=1.2):
#         super().__init__()
#         self.register_buffer('A_base', A_init)
#         self.W = nn.Parameter(torch.ones(M, L))
#         self.clamp_min = clamp_min
#         self.clamp_max = clamp_max

#     def get_calibrated_A(self):
#         w_clamped = torch.clamp(self.W, self.clamp_min, self.clamp_max)
#         return self.A_base * w_clamped

# # ==========================================
# # 5. 主网络 (LipidENNet)
# # ==========================================
# class LipidENNet(nn.Module):
#     def __init__(self, A_init, K=10, clamp_min=0.8, clamp_max=1.2):
#         super().__init__()
#         M, L = A_init.shape
#         self.calibrator = SpectralCalibration(M, L, A_init, clamp_min, clamp_max)
#         self.layers = nn.ModuleList([ElasticISTABlock(M, L) for _ in range(K)])
#         self.init_scale = nn.Parameter(torch.tensor(5.0))

#     def forward(self, B_obs):
#         A_curr = self.calibrator.get_calibrated_A()
        
#         X = torch.einsum('ml, bmhw -> blhw', A_curr, B_obs) * self.init_scale
#         X = F.relu(X)
        
#         X_prev = None
#         X_layers = []
#         masks = []
        
#         for layer in self.layers:
#             X_next, mask = layer(X, X_prev, B_obs, A_curr)
#             X_prev = X
#             X = X_next
#             X_layers.append(X)
#             masks.append(mask)
            
#         return X, X_layers, A_curr, masks
        

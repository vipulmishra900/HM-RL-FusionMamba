import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionMamba.mamba_block import MambaBlock

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Performs local 1D convolution along channels after global average pooling.
    """
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        # Calculate adaptive kernel size k based on channel dimension
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 == 1 else t + 1
        k_size = max(k_size, 3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class LDC(nn.Module):
    """
    Learnable Descriptive Convolution (LDC) module (Eq. 5).
    Dynamically modulates the kernel weights using a learnable mask and center mask.
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, groups=channels, bias=bias
        )
        
        # Center mask for difference calculation
        center_mask = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        self.register_buffer('center_mask', center_mask)
        
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]), requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        # weight shape: (channels, 1, 3, 3) since groups=channels
        # weight_sum shape: (channels, 1, 1, 1)
        weight_sum = self.conv.weight.sum(dim=(-1, -2), keepdim=True)
        
        # mask shape: (channels, 1, 3, 3)
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask[None, None, :, :] * weight_sum
               
        modulated_weight = self.conv.weight * mask
        return F.conv2d(
            input=x, weight=modulated_weight, bias=self.conv.bias, 
            stride=self.conv.stride, padding=self.conv.padding, 
            dilation=self.conv.dilation, groups=self.conv.groups
        )

class ESSM(nn.Module):
    """
    Efficient State Space Module (ESSM).
    Processes 2D spatial features in 4 scanning directions using a shared Mamba block
    to model 2D spatial context.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = MambaBlock(d_model=channels, expand=2)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        L = H * W
        
        # 1. 4-way scanning representation
        # Direction 1: Row-first forward scan
        x1 = x.flatten(2).transpose(1, 2)  # (B, L, C)
        # Direction 2: Row-first backward scan
        x2 = x1.flip(dims=[1])  # (B, L, C)
        # Direction 3: Column-first forward scan
        x3 = x.transpose(2, 3).flatten(2).transpose(1, 2)  # (B, L, C)
        # Direction 4: Column-first backward scan
        x4 = x3.flip(dims=[1])  # (B, L, C)
        
        # Group scan sequences along batch dimension for parallel Mamba execution
        combined = torch.cat([x1, x2, x3, x4], dim=0)  # (4*B, L, C)
        combined_norm = self.norm(combined)
        
        # 2. Process via selective scan (MambaBlock)
        out_combined = self.mamba(combined_norm)  # (4*B, L, C)
        
        # Split back to individual directions
        y1_flat, y2_flat, y3_flat, y4_flat = out_combined.chunk(4, dim=0)
        
        # 3. Reshape and invert scans
        y1 = y1_flat.transpose(1, 2).view(B, C, H, W)
        y2 = y2_flat.flip(dims=[1]).transpose(1, 2).view(B, C, H, W)
        y3 = y3_flat.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        y4 = y4_flat.flip(dims=[1]).transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        
        # 4. Sum/average direction representations
        out = (y1 + y2 + y3 + y4) / 4.0
        return out

class DVSSBlock(nn.Module):
    """
    Dynamic Vision State Space (DVSS) block.
    Integrates LayerNorm, ESSM, LDC, and ECA in a serial pipeline:
    F^{n+1} = ECA(LDC(ESSM(LN(F^n)))) + F^n
    """
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.GroupNorm(num_groups=1, num_channels=channels)  # Equivalent to LayerNorm over channels/pixels
        self.essm = ESSM(channels)
        self.ldc = LDC(channels)
        self.eca = ECA(channels)

    def forward(self, x):
        # 1. Normalize
        x_norm = self.ln(x)
        
        # 2. Global spatial context path (ESSM)
        out_global = self.essm(x_norm)
        
        # 3. Local neighborhood texture path (LDC)
        out_local = self.ldc(out_global)
        
        # 4. Channel selection and enhancement (ECA)
        out = self.eca(out_local)
        
        # 5. Residual connection
        return x + out


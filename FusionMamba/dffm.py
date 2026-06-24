import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionMamba.mamba_block import MambaBlock
from FusionMamba.dvss_block import LDC

class DFEM(nn.Module):
    """
    Dynamic Feature Enhancement Module (DFEM) (Eq. 7).
    Enhances features of both modalities by sensing cross-modal texture differences.
    """
    def __init__(self, channels):
        super().__init__()
        self.ldc1 = LDC(channels)
        self.ldc2 = LDC(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2, F_f):
        """
        Args:
            F1 (torch.Tensor): Feature map from modality A (B, C, H, W)
            F2 (torch.Tensor): Feature map from modality B (B, C, H, W)
            F_f (torch.Tensor): Fused cross-modal feature map from CMFM (B, C, H, W)
        Returns:
            D1 (torch.Tensor): Enhanced features for modality A (B, C, H, W)
            D2 (torch.Tensor): Enhanced features for modality B (B, C, H, W)
        """
        # 1. Transform features with Learnable Descriptive Convolutions
        T1 = self.ldc1(F1)
        T2 = self.ldc2(F2)
        
        # 2. Compute difference-sensing weights
        w1 = self.sigmoid(self.gap(T2 - T1))  # (B, C, 1, 1)
        w2 = self.sigmoid(self.gap(T1 - T2))  # (B, C, 1, 1)
        
        # 3. Dynamic feature enhancement
        D1 = F1 + T1 + w1 * F_f
        D2 = F2 + T2 + w2 * F_f
        
        return D1, D2

class CMFM(nn.Module):
    """
    Cross-Modality Fusion Mamba (CMFM) Module.
    Uses Mamba to capture cross-modal correlation dependencies and suppress redundancy.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Projections to align and process features
        self.proj_A = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_B = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Shared Cross-Modal Mamba Block
        self.mamba = MambaBlock(d_model=channels, expand=2)
        self.norm = nn.LayerNorm(channels)
        
        # Cross-modal gating layers
        self.gate_A = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_B = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Out projection
        self.out_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x_A, x_B):
        # x_A, x_B: (B, C, H, W)
        B, C, H, W = x_A.shape
        L = H * W
        
        # 1. Project features
        feat_A = self.proj_A(x_A)
        feat_B = self.proj_B(x_B)
        
        # 2. Flatten and concatenate modalities along the spatial (sequence) dimension
        # Stacking them as [Modality A tokens, Modality B tokens] -> Sequence Length = 2 * L
        flat_A = feat_A.flatten(2).transpose(1, 2)  # (B, L, C)
        flat_B = feat_B.flatten(2).transpose(1, 2)  # (B, L, C)
        
        combined_seq = torch.cat([flat_A, flat_B], dim=1)  # (B, 2 * L, C)
        combined_seq = self.norm(combined_seq)
        
        # 3. Cross-modal Mamba sequence processing
        processed_seq = self.mamba(combined_seq)  # (B, 2 * L, C)
        
        # 4. Split back to individual modalities
        out_flat_A, out_flat_B = torch.split(processed_seq, L, dim=1)
        
        # Reshape back to 2D
        mamba_feats_A = out_flat_A.transpose(1, 2).view(B, C, H, W)
        mamba_feats_B = out_flat_B.transpose(1, 2).view(B, C, H, W)
        
        # 5. Cross-modality mutual gating
        gate_val_A = torch.sigmoid(self.gate_B(mamba_feats_B))
        gate_val_B = torch.sigmoid(self.gate_A(mamba_feats_A))
        
        gated_A = x_A * gate_val_A
        gated_B = x_B * gate_val_B
        
        # 6. Final output fusion
        fused = torch.cat([gated_A, gated_B], dim=1)  # (B, 2 * C, H, W)
        out = self.out_conv(fused)  # (B, C, H, W)
        
        return out

class DFFM(nn.Module):
    """
    Dynamic Feature Fusion Module (DFFM).
    Performs cross-modal fusion via CMFM first, followed by dynamic feature enhancement via DFEM.
    """
    def __init__(self, channels):
        super().__init__()
        self.cmfm = CMFM(channels)
        self.dfem = DFEM(channels)
        self.fuse_proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x_A, x_B):
        """
        Args:
            x_A (torch.Tensor): Feature map from modality A (B, C, H, W)
            x_B (torch.Tensor): Feature map from modality B (B, C, H, W)
        Returns:
            fused (torch.Tensor): Fused multi-modal feature map (B, C, H, W)
            D1 (torch.Tensor): Enhanced modality A features (B, C, H, W)
            D2 (torch.Tensor): Enhanced modality B features (B, C, H, W)
        """
        # 1. Cross-modal Mamba fusion first to get fused feature map
        F_f = self.cmfm(x_A, x_B)
        
        # 2. Dynamic Feature Enhancement of both modalities
        D1, D2 = self.dfem(x_A, x_B, F_f)
        
        # 3. Combine enhanced features for the skip connection output
        fused = self.fuse_proj(torch.cat([D1, D2], dim=1))
        
        return fused, D1, D2


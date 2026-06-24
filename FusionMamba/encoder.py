import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionMamba.dvss_block import DVSSBlock
from FusionMamba.dffm import DFFM

class DownsampleBlock(nn.Module):
    """
    Standard Downsampling Block.
    Reduces spatial resolution (H, W -> H/2, W/2) and projects channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x))) if hasattr(nn.functional, 'relu') else torch.relu(self.norm(self.conv(x)))

class EncoderStage(nn.Module):
    """
    A single stage of the FusionMamba encoder.
    Applies a series of DVSS Blocks to extract modality-specific features.
    """
    def __init__(self, channels, depth):
        super().__init__()
        self.blocks = nn.ModuleList([
            DVSSBlock(channels) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class HierarchicalEncoder(nn.Module):
    """
    Hierarchical dual-stream encoder for multi-modal image fusion.
    Extracts features for Modality A and Modality B, fusing them at each stage.
    Stage depths are pinned to [2, 2, 9, 2].
    """
    def __init__(self, in_channels=1, base_channels=32, depths=[2, 2, 9, 2]):
        super().__init__()
        self.depths = depths
        
        # Initial projection from input channels (e.g., Grayscale/RGB) to base_channels
        self.proj_A = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.proj_B = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.channels = [
            base_channels, 
            base_channels * 2, 
            base_channels * 4, 
            base_channels * 8
        ]
        
        # Encoder Stages for Stream A and B
        self.stages_A = nn.ModuleList()
        self.stages_B = nn.ModuleList()
        
        # Downsamplers to move between stages
        self.downsamplers_A = nn.ModuleList()
        self.downsamplers_B = nn.ModuleList()
        
        # DFFM fusion modules at each stage
        self.fusion_modules = nn.ModuleList()
        
        for i in range(len(depths)):
            # Add Downsamplers for subsequent stages
            if i > 0:
                self.downsamplers_A.append(DownsampleBlock(self.channels[i-1], self.channels[i]))
                self.downsamplers_B.append(DownsampleBlock(self.channels[i-1], self.channels[i]))
            
            # Add DVSS block stages
            self.stages_A.append(EncoderStage(self.channels[i], depths[i]))
            self.stages_B.append(EncoderStage(self.channels[i], depths[i]))
            
            # Add fusion modules
            self.fusion_modules.append(DFFM(self.channels[i]))

    def forward(self, img_A, img_B):
        """
        Args:
            img_A (torch.Tensor): Image input of Modality A (B, in_channels, H, W)
            img_B (torch.Tensor): Image input of Modality B (B, in_channels, H, W)
        Returns:
            fused_features (List[torch.Tensor]): Fused multi-scale feature maps from each stage
        """
        # 1. Initial projections
        feat_A = self.proj_A(img_A)
        feat_B = self.proj_B(img_B)
        
        fused_features = []
        
        # 2. Stage-by-stage forward pass
        for i in range(len(self.depths)):
            # Downsample if we are past stage 1
            if i > 0:
                feat_A = self.downsamplers_A[i-1](feat_A)
                feat_B = self.downsamplers_B[i-1](feat_B)
            
            # Feature extraction
            feat_A = self.stages_A[i](feat_A)
            feat_B = self.stages_B[i](feat_B)
            
            # Multi-modal fusion (unpacks the fused feature map and updated modal features)
            fused_stage, feat_A, feat_B = self.fusion_modules[i](feat_A, feat_B)
            fused_features.append(fused_stage)
            
        return fused_features

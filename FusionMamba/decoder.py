import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionMamba.dvss_block import DVSSBlock

class UpsampleBlock(nn.Module):
    """
    Upsampling block.
    Bilinear interpolation followed by convolution to project channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Double the spatial resolution
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return F.relu(self.norm(self.conv(x_up)))

class DecoderStage(nn.Module):
    """
    A single stage of the FusionMamba decoder.
    Applies DVSS Blocks to reconstruct and refine multi-scale representations.
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

class HierarchicalDecoder(nn.Module):
    """
    Hierarchical reconstruction decoder.
    Progressively upsamples and fuses representations from encoder stages,
    reconstructing the final fused image.
    """
    def __init__(self, out_channels=1, base_channels=32, depths=[2, 9, 2, 2]):
        super().__init__()
        # Symmetrical stage depths in reverse: Stage 4 -> Stage 3 -> Stage 2 -> Stage 1
        self.depths = depths
        
        self.channels = [
            base_channels, 
            base_channels * 2, 
            base_channels * 4, 
            base_channels * 8
        ]
        
        # Upsamplers
        self.upsampler3 = UpsampleBlock(self.channels[3], self.channels[2])
        self.upsampler2 = UpsampleBlock(self.channels[2], self.channels[1])
        self.upsampler1 = UpsampleBlock(self.channels[1], self.channels[0])
        
        # Channel fusion projections (to combine upsampled features with skip connection features)
        self.fuse_proj3 = nn.Conv2d(self.channels[2] * 2, self.channels[2], kernel_size=1)
        self.fuse_proj2 = nn.Conv2d(self.channels[1] * 2, self.channels[1], kernel_size=1)
        self.fuse_proj1 = nn.Conv2d(self.channels[0] * 2, self.channels[0], kernel_size=1)
        
        # Decoder stages
        self.stage4 = DecoderStage(self.channels[3], depths[0])
        self.stage3 = DecoderStage(self.channels[2], depths[1])
        self.stage2 = DecoderStage(self.channels[1], depths[2])
        self.stage1 = DecoderStage(self.channels[0], depths[3])
        
        # Final output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0] // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Bound output pixels to normalized [0, 1] range
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features (List[torch.Tensor]): List of fused features from encoder
                                                 [Stage1, Stage2, Stage3, Stage4]
        Returns:
            fused_image (torch.Tensor): Final reconstructed fused image (B, out_channels, H, W)
        """
        feat1, feat2, feat3, feat4 = fused_features
        
        # 1. Process deepest features (Stage 4)
        out = self.stage4(feat4)
        
        # 2. Upsample and merge with Stage 3 features
        out = self.upsampler3(out)
        out = torch.cat([out, feat3], dim=1)
        out = self.fuse_proj3(out)
        out = self.stage3(out)
        
        # 3. Upsample and merge with Stage 2 features
        out = self.upsampler2(out)
        out = torch.cat([out, feat2], dim=1)
        out = self.fuse_proj2(out)
        out = self.stage2(out)
        
        # 4. Upsample and merge with Stage 1 features
        out = self.upsampler1(out)
        out = torch.cat([out, feat1], dim=1)
        out = self.fuse_proj1(out)
        out = self.stage1(out)
        
        # 5. Output image reconstruction
        fused_image = self.out_conv(out)
        
        return fused_image

import torch
import torch.nn as nn
from FusionMamba.encoder import HierarchicalEncoder
from FusionMamba.decoder import HierarchicalDecoder

class FusionMamba(nn.Module):
    """
    FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba.
    Provides a hierarchical, dual-stream state-space architecture for fusing multi-modal images.
    
    Structure:
      - Encoder Stage Depths: [2, 2, 9, 2]
      - Decoder Stage Depths: [2, 9, 2, 2]
    """
    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        
        # Hierarchical Encoder: extracts features from both streams and fuses them at each stage
        self.encoder = HierarchicalEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depths=[2, 2, 9, 2]
        )
        
        # Hierarchical Decoder: reconstructs the final fused image
        self.decoder = HierarchicalDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            depths=[2, 9, 2, 2]
        )

    def forward(self, img_A, img_B):
        """
        Args:
            img_A (torch.Tensor): Modality A input tensor of shape (B, in_channels, H, W)
            img_B (torch.Tensor): Modality B input tensor of shape (B, in_channels, H, W)
        Returns:
            fused_img (torch.Tensor): Reconstructed fused output tensor of shape (B, out_channels, H, W)
        """
        # 1. Forward through hierarchical encoder to get fused multi-scale features
        fused_features = self.encoder(img_A, img_B)
        
        # 2. Reconstruct the output image via hierarchical decoder
        fused_img = self.decoder(fused_features)
        
        return fused_img

if __name__ == "__main__":
    print("Testing FusionMamba Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = FusionMamba(in_channels=1, base_channels=16, out_channels=1).to(device)
    
    # Create mock inputs (e.g. Infrared and Visible images)
    # Shape: (Batch_size=2, Channels=1, Height=64, Width=64)
    img_A = torch.randn(2, 1, 64, 64).to(device)
    img_B = torch.randn(2, 1, 64, 64).to(device)
    
    print(f"Input A shape: {img_A.shape}")
    print(f"Input B shape: {img_B.shape}")
    
    # Forward pass
    with torch.no_grad():
        fused_output = model(img_A, img_B)
        
    print(f"Output shape:  {fused_output.shape}")
    print("Architecture verified successfully!")

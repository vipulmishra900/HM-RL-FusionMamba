import torch
import torch.nn as nn
from .mamba_block import MambaBlock

class FusionMamba(nn.Module):
    """
    Multi-modal Fusion module utilizing Mamba blocks to fuse heterogeneous
    sensory representations (e.g., visual tokens, tactile/proprioceptive states)
    across sequence/modality steps.
    """
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Linear projections for aligning multimodal features to d_model
        self.visual_align = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.proprio_align = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Modality token embeddings (to identify visual vs. proprioceptive inputs)
        self.modality_embeddings = nn.Parameter(torch.randn(2, d_model))
        
        # Stacked Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Final output layer to map to a joint representation
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, visual_feats, proprio_feats):
        """
        Args:
            visual_feats (torch.Tensor): Visual representations of shape (batch_size, d_model)
            proprio_feats (torch.Tensor): Proprioceptive representations of shape (batch_size, d_model)
        Returns:
            fused_representation (torch.Tensor): Joint state representation of shape (batch_size, d_model)
        """
        batch_size = visual_feats.shape[0]
        
        # Align feature representations
        vis_aligned = self.visual_align(visual_feats).unsqueeze(1)    # (B, 1, d_model)
        prop_aligned = self.proprio_align(proprio_feats).unsqueeze(1)  # (B, 1, d_model)
        
        # Concatenate modalities as a sequence of tokens
        # Sequence dimension represents the different modalities
        tokens = torch.cat([vis_aligned, prop_aligned], dim=1)  # (B, 2, d_model)
        
        # Add modality-specific embeddings
        tokens = tokens + self.modality_embeddings.unsqueeze(0)
        tokens = self.dropout(tokens)
        
        # Forward through Mamba block sequence
        out = tokens
        for layer in self.mamba_layers:
            out = layer(out)
            
        out = self.ln_f(out)
        
        # Global average pool / aggregate over modality tokens
        fused_state = torch.mean(out, dim=1)  # (B, d_model)
        
        # Pass through the fusion projection head
        fused_representation = self.fusion_head(fused_state)
        
        return fused_representation

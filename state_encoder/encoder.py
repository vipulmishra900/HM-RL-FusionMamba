import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    CNN encoder to process visual observations (e.g., RGB images).
    Projects images to a flat latent feature embedding space.
    """
    def __init__(self, input_shape=(3, 64, 64), embed_dim=128):
        super().__init__()
        channels, height, width = input_shape
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),  # Output: 32 x 15 x 15
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        # Output: 64 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),        # Output: 64 x 4 x 4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out_dim = self.conv_net(dummy).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, visual_obs):
        """
        Args:
            visual_obs (torch.Tensor): Visual input of shape (batch_size, C, H, W)
        Returns:
            torch.Tensor: Feature embedding of shape (batch_size, embed_dim)
        """
        features = self.conv_net(visual_obs)
        return self.fc(features)

class ProprioceptiveEncoder(nn.Module):
    """
    MLP encoder to process proprioceptive/tactile observations.
    Projects low-dimensional states to the shared embedding space.
    """
    def __init__(self, input_dim=16, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, proprio_obs):
        """
        Args:
            proprio_obs (torch.Tensor): Proprioceptive input of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Feature embedding of shape (batch_size, embed_dim)
        """
        return self.net(proprio_obs)

class MultiModalEncoder(nn.Module):
    """
    A unified wrapper containing both visual and proprioceptive encoders.
    Processes heterogeneous multi-modal state observations.
    """
    def __init__(self, visual_shape=(3, 64, 64), proprio_dim=16, embed_dim=128):
        super().__init__()
        self.visual_encoder = VisualEncoder(visual_shape, embed_dim)
        self.proprio_encoder = ProprioceptiveEncoder(proprio_dim, embed_dim)
        
    def forward(self, visual_obs, proprio_obs):
        """
        Args:
            visual_obs (torch.Tensor): Image input of shape (batch_size, C, H, W)
            proprio_obs (torch.Tensor): Joint states input of shape (batch_size, D)
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing visual and proprioceptive features
        """
        vis_features = self.visual_encoder(visual_obs)
        proprio_features = self.proprio_encoder(proprio_obs)
        
        return {
            "visual": vis_features,
            "proprio": proprio_features
        }

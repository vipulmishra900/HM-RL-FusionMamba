import torch
from torch.utils.data import Dataset
import numpy as np

class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for Heterogeneous Multi-modal state transitions.
    Yields observations from both visual and proprioceptive/sensor spaces.
    """
    def __init__(self, visual_shape=(3, 64, 64), proprioceptive_dim=16, length=1000):
        super().__init__()
        self.visual_shape = visual_shape
        self.proprioceptive_dim = proprioceptive_dim
        self.length = length
        
        # Simulate some dataset storage
        self.visual_data = np.random.randn(length, *visual_shape).astype(np.float32)
        self.proprioceptive_data = np.random.randn(length, proprioceptive_dim).astype(np.float32)
        self.actions = np.random.randn(length, 6).astype(np.float32)  # Assuming action dim = 6
        self.rewards = np.random.randn(length, 1).astype(np.float32)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        """
        Returns:
            visual_obs (torch.Tensor): Visual image input (C, H, W)
            proprio_obs (torch.Tensor): Sensor / joint configuration (dim,)
            action (torch.Tensor): Executed action
            reward (torch.Tensor): Received reward
        """
        visual_obs = torch.from_numpy(self.visual_data[idx])
        proprio_obs = torch.from_numpy(self.proprioceptive_data[idx])
        action = torch.from_numpy(self.actions[idx])
        reward = torch.from_numpy(self.rewards[idx])
        
        return {
            "visual": visual_obs,
            "proprio": proprio_obs,
            "action": action,
            "reward": reward
        }

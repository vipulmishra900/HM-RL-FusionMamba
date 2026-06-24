import torch
from torch.utils.data import Dataset
from config import HMRLFusionMambaConfig

class RLDatasetAdapter(Dataset):
    """
    Adapter to wrap a PairedImageDataset and expose it in a format
    compatible with the PPO / RL agent observation spaces.
    Converts paired (IR, Visible) outputs into multimodal states.
    """
    def __init__(self, paired_dataset, config=None):
        self.paired_dataset = paired_dataset
        self.config = config if config is not None else HMRLFusionMambaConfig()
        
    def __len__(self):
        return len(self.paired_dataset)
        
    def __getitem__(self, idx):
        paired_item = self.paired_dataset[idx]
        ir_tensor = paired_item["ir"]      # (C, H, W)
        vis_tensor = paired_item["vis"]    # (C, H, W)
        
        # Concat along channel dimension for visual state representation
        visual_obs = torch.cat([vis_tensor, ir_tensor], dim=0)
        
        # Generate placeholder / dummy proprioceptive, action, and reward arrays
        # to match the PPO agent / actor-critic inputs
        proprio_dim = self.config.encoder.proprioceptive_dim
        action_dim = self.config.agent.action_dim
        
        proprio_obs = torch.zeros(proprio_dim, dtype=torch.float32)
        action = torch.zeros(action_dim, dtype=torch.float32)
        reward = torch.zeros(1, dtype=torch.float32)
        
        return {
            "visual": visual_obs,
            "proprio": proprio_obs,
            "action": action,
            "reward": reward,
            "ir_path": paired_item["ir_path"],
            "vis_path": paired_item["vis_path"]
        }

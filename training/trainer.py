import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datasets.custom_dataset import PairedImageDataset
from FusionMamba import FusionMamba
from state_encoder import MultiModalEncoder
from PPO_agent import ActorCritic, PPOAlgorithm
from adaptive_controller import AdaptiveController
from reward_function import MultiModalRewardShaper

class Trainer:
    """
    Supervised trainer class coordinating PairedImageDataset loading from LLVIP,
    FusionMamba model forward pass, custom fusion loss (L1 + SSIM), and optimization.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # 1. Initialize Networks
        self.encoder = MultiModalEncoder(
            visual_shape=config.encoder.visual_input_shape,
            proprio_dim=config.encoder.proprioceptive_dim,
            embed_dim=config.encoder.embed_dim
        ).to(self.device)
        
        # Corrected FusionMamba instantiation to match its actual signature
        self.fusion_mamba = FusionMamba(
            in_channels=1,
            base_channels=32,
            out_channels=1
        ).to(self.device)
        
        self.actor_critic = ActorCritic(
            state_dim=config.mamba.d_model,
            action_dim=config.agent.action_dim,
            hidden_dim=config.agent.hidden_dim,
            action_space_type=config.agent.action_space_type
        ).to(self.device)
        
        # 2. PPO Optimizer (kept for compatibility, not used in supervised loop)
        self.ppo = PPOAlgorithm(self.actor_critic, config)
        
        # 3. Adaptive Controller & Reward Shaper
        self.controller = AdaptiveController(config)
        self.reward_shaper = MultiModalRewardShaper()
        
        # 4. Integrate PairedImageDataset with LLVIP
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        llvip_dir = os.path.join(base_dir, 'datasets', 'LLVIP')
        resize_shape = config.encoder.visual_input_shape[1:]  # (Height, Width)
        
        self.dataset = PairedImageDataset(
            root_dir=llvip_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=True,
            split='train'
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.agent.mini_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 5. Create Adam optimizer for FusionMamba parameters
        self.optimizer = optim.Adam(self.fusion_mamba.parameters(), lr=1e-4)

    def ssim_loss(self, img1, img2, window_size=11):
        import torch.nn.functional as F
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = num / (den + 1e-10)
        return 1.0 - ssim_map.mean()

    def collect_rollouts(self, num_steps):
        """
        Kept for compatibility.
        """
        states_list = []
        actions_list = []
        log_probs_list = []
        returns_list = []
        advantages_list = []
        return states_list, actions_list, log_probs_list, returns_list, advantages_list

    def train(self):
        """
        Executes supervised training iterations for FusionMamba.
        """
        print("Starting supervised FusionMamba training pipeline...")
        self.fusion_mamba.train()
        
        epochs = 1
        
        for epoch in range(1, epochs + 1):
            for iteration, batch in enumerate(self.dataloader, 1):
                ir = batch["ir"].to(self.device)
                vis = batch["vis"].to(self.device)
                
                # Forward pass
                fused = self.fusion_mamba(ir, vis)
                
                # Calculate L1 loss
                loss_l1 = (torch.mean(torch.abs(fused - ir)) + torch.mean(torch.abs(fused - vis))) / 2.0
                
                # Calculate SSIM loss
                loss_ssim = (self.ssim_loss(fused, ir) + self.ssim_loss(fused, vis)) / 2.0
                
                # Total loss: 0.8 * L1 + 0.2 * SSIM
                loss = 0.8 * loss_l1 + 0.2 * loss_ssim
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Print epoch, iteration, and loss
                print(f"Epoch: {epoch} | Iteration: {iteration}/{len(self.dataloader)} | Loss: {loss.item():.6f}")
                    
        print("Supervised training loop completed successfully!")

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
        
        # 4. Integrate PairedImageDataset with LLVIP (configurable path)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if hasattr(config.training, 'dataset_dir') and config.training.dataset_dir:
            llvip_dir = config.training.dataset_dir
        elif os.environ.get("DATASET_DIR"):
            llvip_dir = os.environ.get("DATASET_DIR")
        else:
            llvip_dir = os.path.join(base_dir, 'datasets', 'LLVIP')

        resize_shape = config.encoder.visual_input_shape[1:]  # (Height, Width)
        batch_size = getattr(config.training, 'batch_size', config.agent.mini_batch_size)
        
        # Ensure results directories exist
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        self.dataset = PairedImageDataset(
            root_dir=llvip_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=True,
            split='train'
        )
        if len(self.dataset) == 0:
            print(f"WARNING: No image pairs found in '{llvip_dir}'.")
            print("Please ensure LLVIP is downloaded or provide --dataset-dir <path>.")
            
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(len(self.dataset) > 0),
            num_workers=0
        )
        
        # Integrate validation dataset (LLVIP test split)
        self.val_dataset = PairedImageDataset(
            root_dir=llvip_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=False,
            split='test'
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 5. Create Adam optimizer for FusionMamba parameters
        lr = getattr(config.training, 'lr', 1e-4)
        self.optimizer = optim.Adam(self.fusion_mamba.parameters(), lr=lr)
        
        # 6. Checkpoint initialization
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if hasattr(self.config.training, 'checkpoint_path') and self.config.training.checkpoint_path is not None:
            self.load_checkpoint(self.config.training.checkpoint_path)

    def ssim_loss(self, img1, img2, window_size=11):
        import torch.nn.functional as F
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.clamp(F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq, min=0.0)
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

    def save_checkpoint(self, epoch, path):
        """
        Saves a training checkpoint including model state, optimizer state, epoch, and best_val_loss.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.fusion_mamba.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved successfully to {path}")

    def load_checkpoint(self, path):
        """
        Loads a training checkpoint and restores model state, optimizer state, start_epoch, and best_val_loss.
        """
        if not os.path.exists(path):
            print(f"WARNING: Checkpoint file not found at {path}. Starting from scratch.")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.fusion_mamba.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed training from checkpoint: {path} (restored starting epoch to {self.start_epoch})")

    def validate(self):
        """
        Runs one validation epoch on LLVIP test split and prints/returns average loss.
        """
        if len(self.val_dataloader) == 0:
            print("Validation DataLoader is empty. Skipping validation pass.")
            return float('nan')

        print("Starting validation loop...")
        self.fusion_mamba.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_dataloader, 1):
                ir = batch["ir"].to(self.device)
                vis = batch["vis"].to(self.device)
                
                fused = self.fusion_mamba(ir, vis)
                
                # Calculate L1 loss
                loss_l1 = (torch.mean(torch.abs(fused - ir)) + torch.mean(torch.abs(fused - vis))) / 2.0
                
                # Calculate SSIM loss
                loss_ssim = (self.ssim_loss(fused, ir) + self.ssim_loss(fused, vis)) / 2.0
                
                # Total loss
                loss = 0.8 * loss_l1 + 0.2 * loss_ssim
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        self.fusion_mamba.train()
        return avg_val_loss

    def train(self):
        """
        Executes supervised training iterations for FusionMamba.
        """
        if len(self.dataloader) == 0:
            raise RuntimeError(
                f"Training DataLoader is empty. No image pairs found in '{self.dataset.root_dir}'. "
                "Please verify the dataset path and ensure paired Infrared and Visible images are present."
            )

        print("Starting supervised FusionMamba training pipeline...")
        self.fusion_mamba.train()
        
        total_epochs = getattr(self.config.training, 'epochs', 5)
        max_iterations = getattr(self.config.training, 'max_iterations', None)
        global_step = 0
        
        for epoch in range(self.start_epoch, total_epochs + 1):
            total_train_loss = 0.0
            
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
                
                total_train_loss += loss.item()
                global_step += 1
                
                # Print epoch, iteration, and loss
                print(f"Epoch: {epoch} | Iteration: {iteration}/{len(self.dataloader)} | Loss: {loss.item():.6f}")
                
                if max_iterations is not None and global_step >= max_iterations:
                    print(f"Reached max iterations limit ({max_iterations}). Stopping training.")
                    break
            
            if max_iterations is not None and global_step >= max_iterations:
                break
            
            avg_train_loss = total_train_loss / len(self.dataloader)
            print(f"Epoch {epoch} Training Completed. Average Train Loss: {avg_train_loss:.6f}")
            
            # Run validation pass at the end of the epoch
            val_loss = self.validate()
            print(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save best model
            if not np.isnan(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config.training.checkpoint_dir, "best_model.pth")
                self.save_checkpoint(epoch, best_path)
                print(f"New best model found at epoch {epoch} with validation loss {val_loss:.6f}!")
            
            # Save periodic/latest checkpoint
            latest_path = os.path.join(self.config.training.checkpoint_dir, "latest_checkpoint.pth")
            self.save_checkpoint(epoch, latest_path)
                    
        print("Supervised training loop completed successfully!")

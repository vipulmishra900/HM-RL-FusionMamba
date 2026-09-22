import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class EncoderConfig:
    """Configuration for state encoders processing multi-modal inputs."""
    visual_input_shape: Tuple[int, int, int] = (3, 64, 64)  # (Channels, Height, Width)
    proprioceptive_dim: int = 16
    joint_features_dim: int = 64
    embed_dim: int = 128  # Shared embedding dimension for fusion

@dataclass
class FusionMambaConfig:
    """Configuration for the FusionMamba fusion module and SSM blocks."""
    # 2D Image Fusion Architecture settings
    in_channels: int = 1    # Grayscale infrared/visible channels
    base_channels: int = 32 # Base feature channels (8.47M parameter configuration)
    out_channels: int = 1   # Fused output channels
    # 1D Token Fusion Module settings (PPO / RL state fusion)
    d_model: int = 128      # Hidden dimension of Mamba block
    d_state: int = 16       # State dimension (SSM parameters)
    d_conv: int = 4         # Convolution kernel size in Mamba block
    expand: int = 2         # Expansion factor
    num_layers: int = 2     # Number of stacked FusionMamba layers
    dropout: float = 0.1    # Dropout probability

@dataclass
class PPOAgentConfig:
    """Configuration for PPO agent and Actor-Critic policies."""
    action_space_type: str = "continuous"  # "continuous" or "discrete"
    action_dim: int = 6                    # Dimension of control action space
    hidden_dim: int = 256                  # Size of MLP heads
    lr_actor: float = 3e-4                 # Learning rate for actor network
    lr_critic: float = 1e-3                # Learning rate for critic network
    gamma: float = 0.99                    # Discount factor
    gae_lambda: float = 0.95               # GAE parameter
    clip_epsilon: float = 0.2              # PPO policy clipping parameter
    ppo_epochs: int = 10                   # Number of updates per iteration
    mini_batch_size: int = 64              # Batch size for policy updates
    entropy_coef: float = 0.01             # Entropy coefficient for exploration
    value_loss_coef: float = 0.5           # Critic value loss coefficient
    max_grad_norm: float = 0.5             # Gradient clipping threshold

@dataclass
class AdaptiveControllerConfig:
    """Configuration for the adaptive controller layer."""
    control_mode: str = "safety_filter"    # Options: "safety_filter", "direct", "pid_hybrid"
    kp: float = 1.0                        # Proportional gain (if hybrid PID)
    kd: float = 0.1                        # Derivative gain (if hybrid PID)
    safety_boundary: float = 2.0           # Action boundaries or threshold limits

@dataclass
class TrainingConfig:
    """Configuration for the training loop and environment simulation."""
    env_name: str = "CustomHMEnv-v0"
    num_envs: int = 4
    total_timesteps: int = 1_000_000
    num_steps: int = 2048                  # Steps collected per environment per iteration
    seed: int = 42
    log_dir: str = "results/logs"
    checkpoint_dir: str = "results/checkpoints"
    save_interval: int = 10                 # Save model every N training iterations
    eval_interval: int = 5                  # Evaluate model every N training iterations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-selects CUDA if available
    checkpoint_path: str = None
    dataset_dir: str = None                 # Optional custom dataset path (e.g., Kaggle /kaggle/input/...)
    epochs: int = 5                         # Supervised training epochs
    batch_size: int = 64                    # Batch size for supervised DataLoader
    lr: float = 1e-4                        # Learning rate for FusionMamba optimizer


@dataclass
class HMRLFusionMambaConfig:
    """Master configuration class grouping all sub-configs."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    mamba: FusionMambaConfig = field(default_factory=FusionMambaConfig)
    agent: PPOAgentConfig = field(default_factory=PPOAgentConfig)
    controller: AdaptiveControllerConfig = field(default_factory=AdaptiveControllerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Create directories if they do not exist
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)

if __name__ == "__main__":
    # Quick configuration instantiation test
    config = HMRLFusionMambaConfig()
    print("Configuration loaded successfully!")
    print(f"Device: {config.training.device}")
    print(f"Action Dimension: {config.agent.action_dim}")

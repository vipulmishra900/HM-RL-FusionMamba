import torch
import numpy as np
from state_encoder import MultiModalEncoder
from FusionMamba import FusionMamba
from PPO_agent import ActorCritic, PPOAlgorithm
from adaptive_controller import AdaptiveController
from reward_function import MultiModalRewardShaper

class Trainer:
    """
    Main trainer class coordinating state encoding, FusionMamba, PPO policy optimization,
    adaptive controller boundaries, and environment rollouts.
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
        
        self.fusion_mamba = FusionMamba(
            d_model=config.mamba.d_model,
            d_state=config.mamba.d_state,
            d_conv=config.mamba.d_conv,
            expand=config.mamba.expand,
            num_layers=config.mamba.num_layers,
            dropout=config.mamba.dropout
        ).to(self.device)
        
        self.actor_critic = ActorCritic(
            state_dim=config.mamba.d_model,
            action_dim=config.agent.action_dim,
            hidden_dim=config.agent.hidden_dim,
            action_space_type=config.agent.action_space_type
        ).to(self.device)
        
        # 2. PPO Optimizer
        self.ppo = PPOAlgorithm(self.actor_critic, config)
        
        # 3. Adaptive Controller & Reward Shaper
        self.controller = AdaptiveController(config)
        self.reward_shaper = MultiModalRewardShaper()
        
    def collect_rollouts(self, num_steps):
        """
        Simulates interacting with the environment to collect trajectories.
        For skeleton purposes, this simulates environment step responses.
        """
        states_list = []
        actions_list = []
        log_probs_list = []
        returns_list = []
        advantages_list = []
        
        # Dummy loop simulating interaction steps
        # In actual environment: obs = env.reset(), etc.
        for _ in range(num_steps):
            # 1. Simulate multi-modal observations
            vis_obs = torch.randn(1, *self.config.encoder.visual_input_shape, device=self.device)
            proprio_obs = torch.randn(1, self.config.encoder.proprioceptive_dim, device=self.device)
            
            # 2. Forward through encoders
            feats = self.encoder(vis_obs, proprio_obs)
            
            # 3. Fuse representations via Mamba
            fused_state = self.fusion_mamba(feats["visual"], feats["proprio"])  # Shape: (1, d_model)
            
            # 4. Forward through ActorCritic policy
            dist, value = self.actor_critic(fused_state)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if self.config.agent.action_space_type == "continuous":
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                
            # 5. Apply adaptive control safety filter
            filtered_action = self.controller.filter_action(action)
            
            # 6. Simulate reward computation
            # Assume target_state is zero vector for tracking simulation
            prev_act = np.zeros(self.config.agent.action_dim)
            step_reward, reward_breakdown = self.reward_shaper.compute_reward(
                state=proprio_obs.cpu().numpy()[0],
                action=filtered_action.cpu().numpy()[0],
                prev_action=prev_act,
                target_state=np.zeros(self.config.encoder.proprioceptive_dim)
            )
            
            # Store values
            states_list.append(fused_state.squeeze(0))
            actions_list.append(action.squeeze(0))
            log_probs_list.append(log_prob.squeeze(0))
            
            # Simulated return & advantage calculations (e.g. Monte-Carlo or TD error)
            returns_list.append(torch.tensor([step_reward], device=self.device))
            advantages_list.append(torch.tensor([step_reward - value.item()], device=self.device))
            
        # Stack all lists to tensors
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)
        log_probs = torch.stack(log_probs_list)
        returns = torch.stack(returns_list)
        advantages = torch.stack(advantages_list)
        
        return states, actions, log_probs, returns, advantages

    def train(self):
        """
        Executes PPO training iterations.
        """
        print("Starting HM-RL-FusionMamba training pipeline...")
        print(f"Total target timesteps: {self.config.training.total_timesteps}")
        
        iterations = 5  # Scaled down for skeleton verification runs
        for i in range(1, iterations + 1):
            # Collect environment interaction samples
            states, actions, log_probs, returns, advantages = self.collect_rollouts(
                self.config.training.num_steps // 10  # Scaled down for verification speed
            )
            
            # Update policy using collected rollouts
            metrics = self.ppo.update(states, actions, log_probs, returns, advantages)
            
            print(f"Iteration {i}/{iterations} | "
                  f"Actor Loss: {metrics['actor_loss']:.4f} | "
                  f"Critic Loss: {metrics['critic_loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f} | "
                  f"Total Loss: {metrics['total_loss']:.4f}")
            
        print("Training loop simulation completed successfully!")

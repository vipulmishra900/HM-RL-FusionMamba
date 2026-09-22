import torch
import torch.nn as nn
import torch.optim as optim

class PPOAlgorithm:
    """
    Implements Proximal Policy Optimization (PPO) clip update logic.
    Ref: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017).
    """
    def __init__(self, actor_critic, config):
        self.actor_critic = actor_critic
        self.config = config.agent
        
        # Optimizers
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor_mean.parameters() if hasattr(self.actor_critic, 'actor_mean') else self.actor_critic.actor_logits.parameters(), 'lr': self.config.lr_actor},
            {'params': self.actor_critic.critic.parameters(), 'lr': self.config.lr_critic}
        ])
        
        # Add actor_logstd to actor parameters if continuous
        if hasattr(self.actor_critic, 'actor_logstd'):
            self.optimizer.add_param_group({'params': [self.actor_critic.actor_logstd], 'lr': self.config.lr_actor})
            
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Executes a policy/value function update step using PPO.
        
        Args:
            states (torch.Tensor): States of shape (N, state_dim)
            actions (torch.Tensor): Taken actions of shape (N, action_dim)
            old_log_probs (torch.Tensor): Log probabilities under the old policy of shape (N, 1)
            returns (torch.Tensor): Target returns for value function of shape (N, 1)
            advantages (torch.Tensor): Calculated advantages of shape (N, 1)
        Returns:
            metrics (Dict[str, float]): Update metrics (actor_loss, critic_loss, entropy, total_loss)
        """
        num_samples = states.shape[0]
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        epoch_total_loss = 0
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(num_samples, device=states.device)
            
            for start_idx in range(0, num_samples, self.config.mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + self.config.mini_batch_size]
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                
                # Evaluate actions under current policy
                new_log_probs, entropy, state_values = self.actor_critic.evaluate_actions(b_states, b_actions)
                
                # Probability ratio: r_t(theta) = pi_theta(a|s) / pi_old(a|s)
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                
                # Surrogate losses
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * b_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (MSE loss)
                critic_loss = nn.functional.mse_loss(state_values, b_returns)
                
                # Total loss = actor_loss + c1 * critic_loss - c2 * entropy
                total_loss = (
                    actor_loss 
                    + self.config.value_loss_coef * critic_loss 
                    - self.config.entropy_coef * entropy.mean()
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Record metrics
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_total_loss += total_loss.item()
                
        num_updates = self.config.ppo_epochs * (num_samples // self.config.mini_batch_size + 1)
        
        return {
            "actor_loss": epoch_actor_loss / num_updates,
            "critic_loss": epoch_critic_loss / num_updates,
            "entropy": epoch_entropy / num_updates,
            "total_loss": epoch_total_loss / num_updates
        }

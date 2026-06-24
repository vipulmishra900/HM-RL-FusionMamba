import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class ActorCritic(nn.Module):
    """
    Unified Actor-Critic architecture that processes a fused state representation.
    Supports continuous and discrete action spaces.
    """
    def __init__(self, state_dim=128, action_dim=6, hidden_dim=256, action_space_type="continuous"):
        super().__init__()
        self.action_space_type = action_space_type
        
        # Shared or individual feature projection
        # Actor network head
        if self.action_space_type == "continuous":
            self.actor_mean = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()  # Bounds continuous action means to [-1, 1]
            )
            # Log standard deviation parameter for action exploration
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor_logits = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
        # Critic network head (predicts state value V(s))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Calculates actions and state values.
        """
        # Get critic state value
        value = self.critic(state)
        
        if self.action_space_type == "continuous":
            mean = self.actor_mean(state)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            dist = Normal(mean, std)
        else:
            logits = self.actor_logits(state)
            dist = Categorical(logits=logits)
            
        return dist, value
        
    def evaluate_actions(self, state, action):
        """
        Evaluates the log probability, entropy, and value for a given action.
        Useful during policy update steps.
        """
        dist, value = self.forward(state)
        
        log_prob = dist.log_prob(action)
        if self.action_space_type == "continuous":
            # Sum log probabilities over action dimension
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            log_prob = log_prob.unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
            
        return log_prob, entropy, value

import numpy as np
import torch

class MultiModalRewardShaper:
    """
    Computes shaped rewards for Heterogeneous Multi-modal RL tasks.
    Encapsulates reward components to support publication-grade multi-objective analysis.
    """
    def __init__(self, tracking_weight=1.0, smoothness_weight=0.1, energy_weight=0.01):
        self.w_track = tracking_weight
        self.w_smooth = smoothness_weight
        self.w_energy = energy_weight
        
    def compute_reward(self, state, action, prev_action, target_state):
        """
        Computes structured reward.
        
        Args:
            state (np.ndarray): Current system states
            action (np.ndarray): Executed action vector
            prev_action (np.ndarray): Previously executed action vector
            target_state (np.ndarray): Reference trajectory/target states
        Returns:
            total_reward (float): Shaped reward value
            info (dict): Breakdowns of reward components (for analysis/plotting)
        """
        # 1. Tracking reward (negative distance to target)
        tracking_err = np.linalg.norm(state - target_state)
        r_track = -tracking_err
        
        # 2. Action smoothness penalty (penalize sudden action changes)
        action_diff = np.linalg.norm(action - prev_action)
        r_smooth = -action_diff
        
        # 3. Energy/Control effort penalty
        r_energy = -np.linalg.norm(action)
        
        # Weighted combination
        total_reward = (
            self.w_track * r_track 
            + self.w_smooth * r_smooth 
            + self.w_energy * r_energy
        )
        
        return float(total_reward), {
            "reward_tracking": float(r_track),
            "reward_smoothness": float(r_smooth),
            "reward_energy": float(r_energy)
        }

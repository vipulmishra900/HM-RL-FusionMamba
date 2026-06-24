import torch
import numpy as np

class AdaptiveController:
    """
    An Adaptive Controller integration layer.
    Combines high-level reinforcement learning actions with low-level
    control laws, joint constraints, or safety-filter boundaries.
    """
    def __init__(self, config):
        self.config = config.controller
        
    def filter_action(self, rl_action, current_state=None):
        """
        Applies control bounds or safety filters to raw RL actions.
        
        Args:
            rl_action (torch.Tensor or np.ndarray): Raw continuous/discrete actions from policy
            current_state (dict): Optional sensor states to compute dynamic limits (e.g. joint velocities)
        Returns:
            filtered_action: Safe control actions matching original format
        """
        # Convert torch.Tensor to numpy if necessary for low-level interaction
        is_tensor = isinstance(rl_action, torch.Tensor)
        if is_tensor:
            device = rl_action.device
            action_np = rl_action.detach().cpu().numpy()
        else:
            action_np = np.array(rl_action)
            
        if self.config.control_mode == "safety_filter":
            # Clip raw action within safe physical boundaries
            filtered_action = np.clip(
                action_np, 
                -self.config.safety_boundary, 
                self.config.safety_boundary
            )
        elif self.config.control_mode == "pid_hybrid":
            # Combine RL action with proportional-derivative feedback error correction
            # (Assuming current_state provides error tracking fields in real settings)
            error = current_state.get("error", np.zeros_like(action_np)) if current_state else np.zeros_like(action_np)
            deriv = current_state.get("deriv_error", np.zeros_like(action_np)) if current_state else np.zeros_like(action_np)
            
            feedback = self.config.kp * error + self.config.kd * deriv
            filtered_action = action_np + feedback
        else:
            # Direct pass-through
            filtered_action = action_np
            
        if is_tensor:
            return torch.from_numpy(filtered_action).to(device)
        return filtered_action

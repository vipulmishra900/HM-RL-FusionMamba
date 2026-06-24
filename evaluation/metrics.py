import numpy as np

def compute_tracking_rmse(states, targets):
    """
    Computes Root Mean Square Error (RMSE) between states and reference targets.
    
    Args:
        states (np.ndarray): Actual states trajectory of shape (N, dim)
        targets (np.ndarray): Target states trajectory of shape (N, dim)
    Returns:
        float: RMSE value
    """
    error = states - targets
    mse = np.mean(error ** 2)
    return float(np.sqrt(mse))

def compute_control_effort(actions):
    """
    Computes Control Effort (RMS of action signals).
    High control effort indicates high energy usage or erratic movements.
    
    Args:
        actions (np.ndarray): Action sequence of shape (N, action_dim)
    Returns:
        float: Control effort value
    """
    squared_actions = actions ** 2
    mean_effort = np.mean(squared_actions)
    return float(np.sqrt(mean_effort))

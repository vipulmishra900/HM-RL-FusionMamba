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

def _to_gray_numpy(image):
    """
    Converts PyTorch tensor or numpy array to a 2D numpy grayscale array.
    """
    if hasattr(image, "detach"):  # Torch tensor
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)
    
    # Remove batch dimension if it exists and has size 1
    if img_np.ndim == 4:
        if img_np.shape[0] == 1:
            img_np = img_np[0]
        else:
            raise ValueError(f"Batch size must be 1 for evaluation, but got shape {img_np.shape}")
            
    # Handle channel configurations
    if img_np.ndim == 3:
        # Check if shape is (C, H, W) or (H, W, C)
        if img_np.shape[0] in [1, 3]:  # C is first
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            else:
                img_np = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        elif img_np.shape[-1] in [1, 3]:  # C is last
            if img_np.shape[-1] == 1:
                img_np = img_np[:, :, 0]
            else:
                img_np = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
        else:
            if 1 in img_np.shape:
                img_np = np.squeeze(img_np)
            else:
                raise ValueError(f"Unsupported image shape: {img_np.shape}")
                
    if img_np.ndim > 2:
        img_np = np.squeeze(img_np)
        
    return img_np

def entropy_metric(image):
    """
    Computes the information entropy of an image.
    
    Args:
        image (np.ndarray or torch.Tensor): Input grayscale or color image.
    Returns:
        float: Entropy value.
    """
    img = _to_gray_numpy(image)
    if img.size == 0:
        return 0.0
    if img.max() <= 1.01 and img.min() >= 0.0:
        img = img * 255.0
    img_int = np.clip(np.round(img), 0, 255).astype(np.uint8)
    
    hist, _ = np.histogram(img_int, bins=256, range=(0, 255))
    p = hist / float(hist.sum() + 1e-10)
    p_nz = p[p > 0]
    
    entropy = -np.sum(p_nz * np.log2(p_nz))
    return float(entropy)

def spatial_frequency(image):
    """
    Computes the spatial frequency of an image.
    
    Args:
        image (np.ndarray or torch.Tensor): Input grayscale or color image.
    Returns:
        float: Spatial frequency value.
    """
    img = _to_gray_numpy(image)
    if img.size == 0:
        return 0.0
    if img.max() <= 1.01 and img.min() >= 0.0:
        img = img * 255.0
        
    rf = np.diff(img, axis=1)
    cf = np.diff(img, axis=0)
    
    rf_val = np.sum(rf**2) / float(img.size)
    cf_val = np.sum(cf**2) / float(img.size)
    
    sf = np.sqrt(rf_val + cf_val + 1e-10)
    return float(sf)

def standard_deviation_metric(image):
    """
    Computes the standard deviation (contrast measure) of an image.
    
    Args:
        image (np.ndarray or torch.Tensor): Input grayscale or color image.
    Returns:
        float: Standard deviation.
    """
    img = _to_gray_numpy(image)
    if img.size == 0:
        return 0.0
    if img.max() <= 1.01 and img.min() >= 0.0:
        img = img * 255.0
    return float(np.std(img))

def average_gradient(image):
    """
    Computes the average gradient (sharpness/clarity) of an image.
    
    Args:
        image (np.ndarray or torch.Tensor): Input grayscale or color image.
    Returns:
        float: Average gradient.
    """
    img = _to_gray_numpy(image)
    if img.size == 0 or img.shape[0] <= 1 or img.shape[1] <= 1:
        return 0.0
    if img.max() <= 1.01 and img.min() >= 0.0:
        img = img * 255.0
        
    dx = img[1:, :-1] - img[:-1, :-1]
    dy = img[:-1, 1:] - img[:-1, :-1]
    
    grad_mag = np.sqrt((dx**2 + dy**2) / 2.0 + 1e-10)
    return float(np.mean(grad_mag))

def _mutual_information(img1, img2):
    img1_gray = _to_gray_numpy(img1)
    img2_gray = _to_gray_numpy(img2)
    
    if img1_gray.max() <= 1.01:
        img1_gray = img1_gray * 255.0
    if img2_gray.max() <= 1.01:
        img2_gray = img2_gray * 255.0
        
    img1_int = np.clip(np.round(img1_gray), 0, 255).astype(np.uint8)
    img2_int = np.clip(np.round(img2_gray), 0, 255).astype(np.uint8)
    
    hist_2d, _, _ = np.histogram2d(img1_int.ravel(), img2_int.ravel(), bins=256, range=[[0, 255], [0, 255]])
    p_xy = hist_2d / (np.sum(hist_2d) + 1e-10)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    p_xy_nz = p_xy[p_xy > 0]
    p_x_nz = p_x[p_x > 0]
    p_y_nz = p_y[p_y > 0]
    
    h_x = -np.sum(p_x_nz * np.log2(p_x_nz))
    h_y = -np.sum(p_y_nz * np.log2(p_y_nz))
    h_xy = -np.sum(p_xy_nz * np.log2(p_xy_nz))
    
    return float(h_x + h_y - h_xy)

def mutual_information_metric(fused, ir, vis):
    """
    Computes the mutual information metric for image fusion: MI(fused, ir) + MI(fused, vis).
    
    Args:
        fused (np.ndarray or torch.Tensor): Fused image.
        ir (np.ndarray or torch.Tensor): Infrared source image.
        vis (np.ndarray or torch.Tensor): Visible source image.
    Returns:
        float: Mutual information metric score.
    """
    mi_ir = _mutual_information(fused, ir)
    mi_vis = _mutual_information(fused, vis)
    return float(mi_ir + mi_vis)

def vifp_mscale(ref, dist):
    import scipy.ndimage
    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2**(4 - scale + 1) + 1
        sd = N / 5.0
        if scale > 1:
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
        
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0
        
        g[sigma2_sq < eps] = 0
        sv_sq[sv_sq <= eps] = eps
        
        num += np.sum(np.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))
        
    if abs(den) < eps:
        return 0.0
    vifp_val = num / den
    if np.isnan(vifp_val) or np.isinf(vifp_val):
        return 0.0
    return float(vifp_val)

def vif_metric(fused, ir, vis):
    """
    Computes the Visual Information Fidelity (VIF) metric for image fusion.
    
    Args:
        fused (np.ndarray or torch.Tensor): Fused image.
        ir (np.ndarray or torch.Tensor): Infrared source image.
        vis (np.ndarray or torch.Tensor): Visible source image.
    Returns:
        float: VIF metric score.
    """
    fused_np = _to_gray_numpy(fused)
    ir_np = _to_gray_numpy(ir)
    vis_np = _to_gray_numpy(vis)
    
    if fused_np.max() <= 1.01:
        fused_np = fused_np * 255.0
    if ir_np.max() <= 1.01:
        ir_np = ir_np * 255.0
    if vis_np.max() <= 1.01:
        vis_np = vis_np * 255.0
        
    vif_ir = vifp_mscale(ir_np, fused_np)
    vif_vis = vifp_mscale(vis_np, fused_np)
    
    return float((vif_ir + vif_vis) / 2.0)

def qabf_metric(fused, ir, vis):
    """
    Computes the Qabf (edge preservation) metric for image fusion.
    
    Args:
        fused (np.ndarray or torch.Tensor): Fused image.
        ir (np.ndarray or torch.Tensor): Infrared source image.
        vis (np.ndarray or torch.Tensor): Visible source image.
    Returns:
        float: Qabf score.
    """
    import cv2
    fused_np = _to_gray_numpy(fused)
    ir_np = _to_gray_numpy(ir)
    vis_np = _to_gray_numpy(vis)
    
    if fused_np.max() <= 1.01:
        fused_np = fused_np * 255.0
    if ir_np.max() <= 1.01:
        ir_np = ir_np * 255.0
    if vis_np.max() <= 1.01:
        vis_np = vis_np * 255.0
        
    # Constants
    L = 1.0
    Tg = 0.9994
    kg = -15.0
    Dg = 0.5
    Ta = 0.9879
    ka = -22.0
    Da = 0.8
    
    def get_gradient(img):
        img_64 = img.astype(np.float64)
        gx = cv2.Sobel(img_64, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_64, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(gx**2 + gy**2)
        alpha = np.arctan2(gy, gx)
        return g, alpha

    g_F, a_F = get_gradient(fused_np)
    g_A, a_A = get_gradient(ir_np)
    g_B, a_B = get_gradient(vis_np)
    
    def get_preservation(g_S, a_S, g_F, a_F):
        g_max = np.maximum(g_S, g_F)
        g_min = np.minimum(g_S, g_F)
        
        G = np.zeros_like(g_S)
        mask = g_max > 1e-10
        G[mask] = g_min[mask] / g_max[mask]
        
        diff = np.abs(a_S - a_F)
        diff = np.where(diff > np.pi, 2.0 * np.pi - diff, diff)
        A = np.abs(diff - np.pi / 2.0) / (np.pi / 2.0)
        
        Qg = Tg / (1.0 + np.exp(kg * (G - Dg)))
        Qa = Ta / (1.0 + np.exp(ka * (A - Da)))
        
        return Qg * Qa
        
    Q_AF = get_preservation(g_A, a_A, g_F, a_F)
    Q_BF = get_preservation(g_B, a_B, g_F, a_F)
    
    w_A = g_A ** L
    w_B = g_B ** L
    
    denom = np.sum(w_A + w_B)
    if denom < 1e-10:
        return 0.0
        
    numerator = np.sum(Q_AF * w_A + Q_BF * w_B)
    q = numerator / denom
    
    if np.isnan(q) or np.isinf(q):
        return 0.0
    return float(q)

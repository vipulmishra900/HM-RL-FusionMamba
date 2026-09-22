import os
import cv2
import numpy as np
try:
    from evaluation.metrics import (
        entropy_metric,
        spatial_frequency,
        standard_deviation_metric,
        average_gradient,
        mutual_information_metric,
        vif_metric,
        qabf_metric
    )
except ImportError:
    from metrics import (
        entropy_metric,
        spatial_frequency,
        standard_deviation_metric,
        average_gradient,
        mutual_information_metric,
        vif_metric,
        qabf_metric
    )

def main():
    # Construct absolute paths relative to this script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ir_path = os.path.join(base_dir, 'results', 'baseline', 'ir_input.png')
    vis_path = os.path.join(base_dir, 'results', 'baseline', 'vis_input.png')
    fused_path = os.path.join(base_dir, 'results', 'baseline', 'fused_output.png')
    
    # Load images
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    vis = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
    fused = cv2.imread(fused_path, cv2.IMREAD_UNCHANGED)
    
    if ir is None:
        raise FileNotFoundError(f"Could not load IR image from {ir_path}")
    if vis is None:
        raise FileNotFoundError(f"Could not load Visible image from {vis_path}")
    if fused is None:
        raise FileNotFoundError(f"Could not load Fused image from {fused_path}")
        
    # Compute all metrics
    en = entropy_metric(fused)
    sf = spatial_frequency(fused)
    sd = standard_deviation_metric(fused)
    ag = average_gradient(fused)
    mi = mutual_information_metric(fused, ir, vis)
    vif = vif_metric(fused, ir, vis)
    qabf = qabf_metric(fused, ir, vis)
    
    # Print the report exactly as requested
    print("====================")
    print("FUSION QUALITY REPORT")
    print("====================")
    print()
    print(f"Entropy: {en:.4f}")
    print(f"Spatial Frequency: {sf:.4f}")
    print(f"Standard Deviation: {sd:.4f}")
    print(f"Average Gradient: {ag:.4f}")
    print(f"Mutual Information: {mi:.4f}")
    print(f"VIF: {vif:.4f}")
    print(f"Qabf: {qabf:.4f}")

if __name__ == "__main__":
    main()

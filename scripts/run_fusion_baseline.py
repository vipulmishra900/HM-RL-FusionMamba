import os
import sys
import time
import torch
from torchvision.utils import save_image

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.custom_dataset import PairedImageDataset
from FusionMamba.fusionmamba import FusionMamba
from config import HMRLFusionMambaConfig

def main():
    # 1. Setup paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llvip_dir = os.path.join(base_path, 'datasets', 'LLVIP')
    output_dir = os.path.join(base_path, 'results', 'baseline')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FusionMamba Image Fusion Baseline Inference Pipeline")
    print("=" * 60)
    
    # 2. Load dataset
    print("Loading LLVIP dataset...")
    # Load config-driven image size
    config = HMRLFusionMambaConfig()
    resize_shape = config.encoder.visual_input_shape[1:]
    
    # For the first baseline run: force image size to 64x64
    resize_shape = (64, 64)
    
    # Support train split by default
    dataset = PairedImageDataset(
        root_dir=llvip_dir,
        resize_shape=resize_shape,
        convert_to_grayscale=True,
        split='train'
    )
    
    if len(dataset) == 0:
        print("ERROR: LLVIP dataset not found or empty.")
        sys.exit(1)
        
    print(f"Dataset successfully loaded. Total items: {len(dataset)}")
    
    # 3. Load one real LLVIP image pair
    item = dataset[0]
    ir_tensor = item["ir"]      # (1, 256, 256)
    vis_tensor = item["vis"]    # (1, 256, 256)
    ir_path = item["ir_path"]
    vis_path = item["vis_path"]
    
    print(f"Loaded image pair:")
    print(f"  - Infrared: {ir_path}")
    print(f"  - Visible:  {vis_path}")
    print(f"  - Tensor shape: {ir_tensor.shape}")
    
    # 4. Initialize model
    print("WARNING: Model is running with randomly initialized weights. Output image is for pipeline verification only, not fusion quality evaluation.")
    model = FusionMamba(in_channels=1, base_channels=32, out_channels=1)
    model.eval()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with base_channels=32. Total parameters: {total_params:,}")
    
    # Prepare batch dimensions
    # Shape: (B=1, C=1, H=64, W=64)
    img_A = ir_tensor.unsqueeze(0)
    img_B = vis_tensor.unsqueeze(0)
    
    # 5. Run FusionMamba forward pass (single run)
    print("Running single forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        fused_tensor = model(img_A, img_B)
    t_single = time.perf_counter() - t0
    
    print(f"Single run CPU inference time: {t_single * 1000:.2f} ms")
    
    # 6. Average inference time over 100 runs
    print("Benchmarking over 100 runs...")
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.perf_counter()
            _ = model(img_A, img_B)
            latencies.append(time.perf_counter() - t_start)
            
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average CPU inference time (100 runs): {avg_latency * 1000:.2f} ms")
    
    # 7. Save outputs
    ir_save_path = os.path.join(output_dir, 'ir_input.png')
    vis_save_path = os.path.join(output_dir, 'vis_input.png')
    fused_save_path = os.path.join(output_dir, 'fused_output.png')
    
    save_image(ir_tensor, ir_save_path)
    save_image(vis_tensor, vis_save_path)
    save_image(fused_tensor.squeeze(0), fused_save_path)
    
    print("-" * 60)
    print("Saved baseline results to:")
    print(f"  - Infrared: {ir_save_path}")
    print(f"  - Visible:  {vis_save_path}")
    print(f"  - Fused:    {fused_save_path}")
    print("=" * 60)
    
    # Output to stdout the exact summary format requested
    print(f"\nMETRICS_REPORT:")
    print(f"Input Shape: {img_A.shape}")
    print(f"Output Shape: {fused_tensor.shape}")
    print(f"Total Parameters: {total_params}")
    print(f"Single Run CPU Inference Time: {t_single * 1000:.2f} ms")
    print(f"Average CPU Inference Time (100 runs): {avg_latency * 1000:.2f} ms")

if __name__ == "__main__":
    main()

import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.custom_dataset import PairedImageDataset
from config import HMRLFusionMambaConfig

def generate_dummy_images(target_dir, num_pairs=8):
    """
    Utility to generate dummy paired images for pipeline testing.
    """
    ir_dir = os.path.join(target_dir, 'infrared')
    vis_dir = os.path.join(target_dir, 'visible')
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Creating {num_pairs} dummy image pairs in {target_dir} for verification...")
    for i in range(num_pairs):
        name = f"dummy_{i:04d}.png"
        # Generate random image arrays
        ir_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        vis_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        # Save as images
        Image.fromarray(ir_array).save(os.path.join(ir_dir, name))
        Image.fromarray(vis_array).save(os.path.join(vis_dir, name))

def delete_dummy_images(target_dir, num_pairs=8):
    """
    Cleans up the dummy images.
    """
    ir_dir = os.path.join(target_dir, 'infrared')
    vis_dir = os.path.join(target_dir, 'visible')
    for i in range(num_pairs):
        name = f"dummy_{i:04d}.png"
        ir_path = os.path.join(ir_dir, name)
        vis_path = os.path.join(vis_dir, name)
        if os.path.exists(ir_path):
            os.remove(ir_path)
        if os.path.exists(vis_path):
            os.remove(vis_path)
    print("Cleaned up dummy verification images.")

def verify_and_print_stats(root_dir, dataset_name):
    """
    Verifies files and basenames for visible and infrared streams in root_dir.
    """
    ir_dir = os.path.join(root_dir, 'infrared')
    vis_dir = os.path.join(root_dir, 'visible')
    
    print(f"\n=== Verification Statistics for {dataset_name} ===")
    
    if not os.path.exists(ir_dir) or not os.path.exists(vis_dir):
        print("Status: DIRECTORIES MISSING OR EMPTY")
        return 0
        
    def discover_images(base_dir):
        images = {}
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, base_dir)
                    rel_key = rel_path.replace('\\', '/')
                    rel_key_no_ext = os.path.splitext(rel_key)[0]
                    images[rel_key_no_ext] = full_path
        return images

    ir_files_dict = discover_images(ir_dir)
    vis_files_dict = discover_images(vis_dir)
    
    ir_files = set(ir_files_dict.keys())
    vis_files = set(vis_files_dict.keys())
    
    common_names = ir_files.intersection(vis_files)
    missing_in_vis = ir_files - vis_files
    missing_in_ir = vis_files - ir_files
    
    print(f"Total Infrared images : {len(ir_files)}")
    print(f"Total Visible images  : {len(vis_files)}")
    print(f"Matched Image Pairs   : {len(common_names)}")
    
    if missing_in_vis:
        print(f"WARNING: {len(missing_in_vis)} IR images have no matching visible image.")
        print(f"First few missing basenames: {list(missing_in_vis)[:5]}")
    if missing_in_ir:
        print(f"WARNING: {len(missing_in_ir)} Visible images have no matching IR image.")
        print(f"First few missing basenames: {list(missing_in_ir)[:5]}")
        
    if len(common_names) == 0:
        print("Status: NO MATCHING PAIRS FOUND")
    elif len(missing_in_vis) == 0 and len(missing_in_ir) == 0:
        print("Status: PERFECTLY ALIGNED (All image files are paired)")
    else:
        print("Status: PARTIALLY ALIGNED (Missing pairs detected)")
        
    return len(common_names)

def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    llvip_dir = os.path.join(root_path, 'datasets', 'LLVIP')
    flir_dir = os.path.join(root_path, 'datasets', 'FLIR')
    
    # 1. Print current real dataset stats
    llvip_pairs = verify_and_print_stats(llvip_dir, "LLVIP")
    flir_pairs = verify_and_print_stats(flir_dir, "FLIR")
    
    # 2. Select target dataset for DataLoader testing
    # If no real images exist, generate dummy images under LLVIP for pipeline testing
    temp_testing = False
    target_dir = llvip_dir
    if llvip_pairs == 0 and flir_pairs == 0:
        print("\nNo real image pairs found. Triggering pipeline verification with dummy images.")
        generate_dummy_images(llvip_dir, num_pairs=8)
        temp_testing = True
        
    # 3. Create Dataset and DataLoader
    print("\nInitializing PairedImageDataset...")
    config = HMRLFusionMambaConfig()
    resize_shape = config.encoder.visual_input_shape[1:]
    print(f"Configured image size from config.py: {resize_shape}")
    
    dataset = PairedImageDataset(
        root_dir=target_dir, 
        resize_shape=resize_shape, 
        convert_to_grayscale=True, 
        is_train=True, 
        val_split=0.25
    )
    
    print(f"Dataset Split (Train): {len(dataset)} items loaded")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 4. Load one batch
    try:
        batch = next(iter(dataloader))
        print("\nDataLoader validation SUCCESSFUL:")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"IR Tensor shape (Batch, Channel, Height, Width): {batch['ir'].shape}")
        print(f"Vis Tensor shape (Batch, Channel, Height, Width): {batch['vis'].shape}")
        print(f"IR Tensor range: [{batch['ir'].min():.4f}, {batch['ir'].max():.4f}]")
        print(f"Vis Tensor range: [{batch['vis'].min():.4f}, {batch['vis'].max():.4f}]")
        
        # Verify shape
        assert batch['ir'].shape[2:] == resize_shape, f"IR Shape mismatch! Expected {resize_shape}, got {batch['ir'].shape[2:]}"
        assert batch['vis'].shape[2:] == resize_shape, f"Vis Shape mismatch! Expected {resize_shape}, got {batch['vis'].shape[2:]}"
        
        # Verify normalization
        assert batch['ir'].min() >= 0.0 and batch['ir'].max() <= 1.0, "IR Tensor normalization failed!"
        assert batch['vis'].min() >= 0.0 and batch['vis'].max() <= 1.0, "Vis Tensor normalization failed!"
        print("Normalization and Shape checks PASSED!")
        
    except Exception as e:
        print(f"\nDataLoader validation FAILED: {str(e)}", file=sys.stderr)
        if temp_testing:
            delete_dummy_images(llvip_dir, num_pairs=8)
        sys.exit(1)
        
    # Cleanup dummy images if generated
    if temp_testing:
        delete_dummy_images(llvip_dir, num_pairs=8)
        
if __name__ == "__main__":
    main()

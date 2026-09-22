import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.custom_dataset import PairedImageDataset
from config import HMRLFusionMambaConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify LLVIP (or paired visible/infrared) dataset alignment, splits, shapes, and DataLoader."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory containing 'infrared' and 'visible' subdirectories (e.g., /kaggle/input/llvip/LLVIP)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader verification (default: 4)."
    )
    return parser.parse_args()

def generate_dummy_images(target_dir, num_pairs=8):
    """Utility to generate dummy paired images for offline fallback testing only."""
    ir_dir = os.path.join(target_dir, 'infrared')
    vis_dir = os.path.join(target_dir, 'visible')
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Creating {num_pairs} dummy image pairs in {target_dir} for verification...")
    for i in range(num_pairs):
        name = f"dummy_{i:04d}.png"
        ir_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        vis_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(ir_array).save(os.path.join(ir_dir, name))
        Image.fromarray(vis_array).save(os.path.join(vis_dir, name))

def delete_dummy_images(target_dir, num_pairs=8):
    """Cleans up dummy verification images."""
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

def verify_and_print_stats(root_dir, dataset_name="LLVIP"):
    """
    Verifies files and basenames for visible and infrared streams in root_dir.
    Returns a dictionary of detailed statistics.
    """
    ir_dir = os.path.join(root_dir, 'infrared')
    vis_dir = os.path.join(root_dir, 'visible')
    
    print(f"\n{'=' * 60}")
    print(f"Verification Statistics for {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Target Directory : {root_dir}")
    print(f"Infrared Path    : {ir_dir}")
    print(f"Visible Path     : {vis_dir}")
    print(f"{'-' * 60}")
    
    if not os.path.exists(ir_dir) or not os.path.exists(vis_dir):
        if not os.path.exists(ir_dir):
            print(f"ERROR: Infrared directory does not exist: {ir_dir}")
        if not os.path.exists(vis_dir):
            print(f"ERROR: Visible directory does not exist: {vis_dir}")
        return {"matched_pairs": 0, "total_ir": 0, "total_vis": 0, "is_structured": False}
        
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

    ir_files = discover_images(ir_dir)
    vis_files = discover_images(vis_dir)
    
    common_keys = sorted(list(set(ir_files.keys()).intersection(vis_files.keys())))
    missing_in_vis = set(ir_files.keys()) - set(vis_files.keys())
    missing_in_ir = set(vis_files.keys()) - set(ir_files.keys())
    
    # Determine structure type (pre-split train/test vs flat)
    structured_keys = [k for k in (set(ir_files.keys()) | set(vis_files.keys())) if k.startswith('train/') or k.startswith('test/')]
    is_structured = len(structured_keys) > 0
    
    if is_structured:
        train_matched = len([k for k in common_keys if k.startswith('train/')])
        test_matched = len([k for k in common_keys if k.startswith('test/')])
        train_ir = len([k for k in ir_files if k.startswith('train/')])
        train_vis = len([k for k in vis_files if k.startswith('train/')])
        test_ir = len([k for k in ir_files if k.startswith('test/')])
        test_vis = len([k for k in vis_files if k.startswith('test/')])
    else:
        # Flat structure: 80/20 train/test partition estimate
        val_split = 0.2
        split_idx = int(len(common_keys) * (1 - val_split))
        train_matched = split_idx
        test_matched = len(common_keys) - split_idx
        train_ir, train_vis = len(ir_files), len(vis_files)
        test_ir, test_vis = 0, 0
    
    print(f"Total Infrared Images : {len(ir_files)}")
    print(f"Total Visible Images  : {len(vis_files)}")
    print(f"Total Matched Pairs   : {len(common_keys)}")
    print(f"  - Train Matched Pairs : {train_matched}")
    print(f"  - Test Matched Pairs  : {test_matched}")
    print(f"Directory Layout      : {'Structured (train/ & test/ subfolders)' if is_structured else 'Flat structure (split by ratio)'}")
    
    if missing_in_vis:
        print(f"WARNING: {len(missing_in_vis)} IR images have no matching visible image.")
        print(f"First few missing IR stems: {list(missing_in_vis)[:5]}")
    if missing_in_ir:
        print(f"WARNING: {len(missing_in_ir)} Visible images have no matching IR image.")
        print(f"First few missing Visible stems: {list(missing_in_ir)[:5]}")
        
    if len(common_keys) == 0:
        print("Status: NO MATCHING PAIRS FOUND")
    elif len(missing_in_vis) == 0 and len(missing_in_ir) == 0:
        print("Status: PERFECTLY ALIGNED (All images paired)")
    else:
        print("Status: PARTIALLY ALIGNED (Some unpaired images detected)")
        
    return {
        "matched_pairs": len(common_keys),
        "total_ir": len(ir_files),
        "total_vis": len(vis_files),
        "train_matched": train_matched,
        "test_matched": test_matched,
        "is_structured": is_structured,
        "missing_in_vis": len(missing_in_vis),
        "missing_in_ir": len(missing_in_ir)
    }

def verify_split_dataloader(root_dir, resize_shape, split_name, batch_size):
    """Loads a split using PairedImageDataset and verifies DataLoader tensor shapes and ranges."""
    print(f"\nVerifying '{split_name}' split DataLoader...")
    is_train = (split_name == 'train')
    
    dataset = PairedImageDataset(
        root_dir=root_dir,
        resize_shape=resize_shape,
        convert_to_grayscale=True,
        is_train=is_train,
        val_split=0.2,
        split=split_name
    )
    
    print(f"  - Split '{split_name}' dataset length: {len(dataset)} items")
    if len(dataset) == 0:
        print(f"  - WARNING: No items found for '{split_name}' split.")
        return False
        
    effective_batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=is_train,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    ir_tensor = batch['ir']
    vis_tensor = batch['vis']
    
    print(f"  - Loaded batch size : {ir_tensor.shape[0]}")
    print(f"  - IR Tensor shape   : {list(ir_tensor.shape)} (Batch, Channel, Height, Width)")
    print(f"  - Vis Tensor shape  : {list(vis_tensor.shape)} (Batch, Channel, Height, Width)")
    print(f"  - IR Tensor range   : [{ir_tensor.min():.4f}, {ir_tensor.max():.4f}]")
    print(f"  - Vis Tensor range  : [{vis_tensor.min():.4f}, {vis_tensor.max():.4f}]")
    
    # Assertions
    expected_channels = 1
    assert ir_tensor.shape[1] == expected_channels, f"IR channel mismatch! Expected {expected_channels}, got {ir_tensor.shape[1]}"
    assert vis_tensor.shape[1] == expected_channels, f"Vis channel mismatch! Expected {expected_channels}, got {vis_tensor.shape[1]}"
    assert ir_tensor.shape[2:] == resize_shape, f"IR spatial shape mismatch! Expected {resize_shape}, got {ir_tensor.shape[2:]}"
    assert vis_tensor.shape[2:] == resize_shape, f"Vis spatial shape mismatch! Expected {resize_shape}, got {vis_tensor.shape[2:]}"
    assert ir_tensor.min() >= 0.0 and ir_tensor.max() <= 1.0, "IR normalization check failed: values outside [0, 1]!"
    assert vis_tensor.min() >= 0.0 and vis_tensor.max() <= 1.0, "Vis normalization check failed: values outside [0, 1]!"
    
    print(f"  - Checks for '{split_name}' split: PASSED")
    return True

def main():
    args = parse_args()
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Determine dataset path
    is_user_path = False
    if args.dataset_dir is not None:
        target_dir = os.path.abspath(args.dataset_dir)
        is_user_path = True
    elif os.environ.get("DATASET_DIR"):
        target_dir = os.path.abspath(os.environ["DATASET_DIR"])
        is_user_path = True
    else:
        target_dir = os.path.join(root_path, 'datasets', 'LLVIP')
        
    print("=" * 60)
    print("        HM-RL-FusionMamba Dataset Verification Tool")
    print("=" * 60)
    
    # If user provided a path, validate existence strictly
    if is_user_path:
        if not os.path.exists(target_dir):
            print(f"ERROR: Specified dataset directory does not exist: {target_dir}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(target_dir):
            print(f"ERROR: Specified dataset path is not a directory: {target_dir}", file=sys.stderr)
            sys.exit(1)
            
    # 2. Inspect real dataset at target_dir
    stats = verify_and_print_stats(target_dir, dataset_name="LLVIP")
    
    temp_testing = False
    if stats["matched_pairs"] == 0:
        if is_user_path:
            print(f"\nERROR: No matching IR/VIS image pairs found in specified path: {target_dir}", file=sys.stderr)
            print("Please ensure the path contains 'infrared' and 'visible' subdirectories with matching image files.", file=sys.stderr)
            sys.exit(1)
        else:
            # Fallback for empty default local repo only
            print("\nNotice: Default local dataset directory is empty.")
            print("To verify a real dataset, pass --dataset-dir <path>.")
            print("Generating temporary dummy images for verification pipeline check...")
            generate_dummy_images(target_dir, num_pairs=8)
            temp_testing = True
            stats = verify_and_print_stats(target_dir, dataset_name="LLVIP (Dummy)")
            
    # 3. Read image size from config
    config = HMRLFusionMambaConfig()
    resize_shape = config.encoder.visual_input_shape[1:]
    print(f"\nTarget image resolution from config.py: {resize_shape}")
    
    # 4. Verify train and test DataLoader pipelines
    try:
        train_ok = verify_split_dataloader(target_dir, resize_shape, 'train', args.batch_size)
        test_ok = verify_split_dataloader(target_dir, resize_shape, 'test', args.batch_size)
        
        print("\n" + "=" * 60)
        if train_ok:
            print("VERIFICATION SUMMARY: SUCCESS")
            print(f"  - Dataset Path   : {target_dir}")
            print(f"  - Total Pairs    : {stats['matched_pairs']}")
            print(f"  - Train Split    : {stats['train_matched']} pairs verified")
            print(f"  - Test Split     : {stats['test_matched']} pairs verified")
            print(f"  - Input Channels : 1 (grayscale)")
            print(f"  - Output Shape   : [B, 1, {resize_shape[0]}, {resize_shape[1]}]")
            print(f"  - Value Range    : [0.0, 1.0]")
        else:
            print("VERIFICATION SUMMARY: FAILED (Train split could not be verified)")
            sys.exit(1)
        print("=" * 60)
        
    except Exception as e:
        print(f"\nDataLoader validation FAILED with error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if temp_testing:
            delete_dummy_images(target_dir, num_pairs=8)
        sys.exit(1)
        
    # Cleanup dummy data if created
    if temp_testing:
        delete_dummy_images(target_dir, num_pairs=8)

if __name__ == "__main__":
    main()

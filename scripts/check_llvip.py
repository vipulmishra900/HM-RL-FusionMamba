import os
import sys

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_llvip_dataset():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llvip_dir = os.path.join(base_path, 'datasets', 'LLVIP')
    
    ir_dir = os.path.join(llvip_dir, 'infrared')
    vis_dir = os.path.join(llvip_dir, 'visible')
    
    print("=" * 60)
    print("LLVIP Dataset Integration Check")
    print("=" * 60)
    print(f"Infrared Path: {ir_dir}")
    print(f"Visible Path:  {vis_dir}")
    print("-" * 60)
    
    if not os.path.exists(ir_dir):
        print(f"ERROR: Infrared directory does not exist: {ir_dir}")
    if not os.path.exists(vis_dir):
        print(f"ERROR: Visible directory does not exist: {vis_dir}")
        
    if not os.path.exists(ir_dir) or not os.path.exists(vis_dir):
        print("\nLLVIP dataset folder structure is incomplete.")
        print("Please follow the instructions in datasets/DOWNLOAD.md to download and place files.")
        return
        
    # Gather files
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
    
    # Calculate matched and mismatched
    common_keys = set(ir_files.keys()).intersection(vis_files.keys())
    
    # Determine structure type
    structured_keys = [k for k in (set(ir_files.keys()) | set(vis_files.keys())) if k.startswith('train/') or k.startswith('test/')]
    is_structured = len(structured_keys) > 0
    
    if is_structured:
        train_ir = len([k for k in ir_files if k.startswith('train/')])
        train_vis = len([k for k in vis_files if k.startswith('train/')])
        train_matched = len([k for k in common_keys if k.startswith('train/')])
        
        test_ir = len([k for k in ir_files if k.startswith('test/')])
        test_vis = len([k for k in vis_files if k.startswith('test/')])
        test_matched = len([k for k in common_keys if k.startswith('test/')])
    else:
        # Backward compatibility with flat directory layout: simulate splits
        import random
        shuffled_keys = sorted(list(common_keys))
        random.seed(42)
        random.shuffle(shuffled_keys)
        split_idx = int(len(shuffled_keys) * 0.8)
        train_matched_keys = set(shuffled_keys[:split_idx])
        test_matched_keys = set(shuffled_keys[split_idx:])
        
        train_ir = len(ir_files) - int(len(ir_files) * 0.2)
        train_vis = len(vis_files) - int(len(vis_files) * 0.2)
        train_matched = len(train_matched_keys)
        
        test_ir = len(ir_files) - train_ir
        test_vis = len(vis_files) - train_vis
        test_matched = len(test_matched_keys)
        
    print(f"Train Infrared Images: {train_ir}")
    print(f"Train Visible Images:  {train_vis}")
    print(f"Train Matched Pairs:   {train_matched}")
    print(f"Test Infrared Images:  {test_ir}")
    print(f"Test Visible Images:   {test_vis}")
    print(f"Test Matched Pairs:    {test_matched}")
    print(f"Total Matched Pairs:   {len(common_keys)}")
    print("-" * 60)
    
    missing_in_vis = ir_files.keys() - vis_files.keys()
    missing_in_ir = vis_files.keys() - ir_files.keys()
    
    if len(ir_files) == 0 and len(vis_files) == 0:
        print("WARNING: Both infrared and visible folders are empty.")
        print("Please extract the LLVIP dataset files into their respective folders.")
    elif len(missing_in_vis) > 0 or len(missing_in_ir) > 0:
        print(f"WARNING: Dataset mismatches detected!")
        if missing_in_vis:
            print(f"  - {len(missing_in_vis)} IR images are missing matching visible counterparts.")
            print(f"    First few missing visible stems: {list(missing_in_vis)[:5]}")
        if missing_in_ir:
            print(f"  - {len(missing_in_ir)} Visible images are missing matching IR counterparts.")
            print(f"    First few missing IR stems: {list(missing_in_ir)[:5]}")
    else:
        print("PERFECT ALIGNMENT: All infrared and visible images are successfully paired.")
        
    print("=" * 60)

if __name__ == "__main__":
    check_llvip_dataset()

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for paired Infrared and Visible images.
    Supports directory structures of datasets like LLVIP, FLIR, and M3FD.
    """
    def __init__(self, root_dir, resize_shape=(64, 64), convert_to_grayscale=True, is_train=True, val_split=0.2, seed=42, split=None):
        super().__init__()
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.convert_to_grayscale = convert_to_grayscale
        self.is_train = is_train
        
        self.ir_dir = os.path.join(root_dir, 'infrared')
        self.vis_dir = os.path.join(root_dir, 'visible')
        
        # Gather images
        if not os.path.exists(self.ir_dir) or not os.path.exists(self.vis_dir):
            self.image_pairs = []
            if not os.path.exists(self.ir_dir):
                print(f"WARNING: Infrared directory is missing: {self.ir_dir}")
            if not os.path.exists(self.vis_dir):
                print(f"WARNING: Visible directory is missing: {self.vis_dir}")
        else:
            # Helper to discover images recursively
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

            ir_files = discover_images(self.ir_dir)
            vis_files = discover_images(self.vis_dir)
            
            # Find common base names (keys) to form pairs
            common_keys = sorted(list(set(ir_files.keys()).intersection(vis_files.keys())))
            
            # Alignment diagnostics
            missing_in_vis = ir_files.keys() - vis_files.keys()
            missing_in_ir = vis_files.keys() - ir_files.keys()
            
            if missing_in_vis:
                print(f"WARNING: {len(missing_in_vis)} infrared files are missing matching visible files.")
            if missing_in_ir:
                print(f"WARNING: {len(missing_in_ir)} visible files are missing matching infrared files.")
            
            print(f"Total matched pairs: {len(common_keys)}")
            
            # Check if there are structured train/test directories
            structured_keys = [k for k in common_keys if k.startswith('train/') or k.startswith('test/')]
            is_structured = len(structured_keys) > 0
            
            # Determine split to use
            if split is None:
                split = 'train' if is_train else 'test'
            
            if split not in ['train', 'test', 'combined']:
                raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'test', 'combined'")
            
            selected_keys = []
            if is_structured:
                # Pre-split structure
                if split == 'train':
                    selected_keys = [k for k in common_keys if k.startswith('train/')]
                elif split == 'test':
                    selected_keys = [k for k in common_keys if k.startswith('test/')]
                elif split == 'combined':
                    selected_keys = common_keys
            else:
                # Flat structure: perform deterministic partition
                if split == 'combined':
                    selected_keys = common_keys
                else:
                    import random
                    shuffled_keys = list(common_keys)
                    random.seed(seed)
                    random.shuffle(shuffled_keys)
                    
                    split_idx = int(len(shuffled_keys) * (1 - val_split))
                    if split == 'train':
                        selected_keys = shuffled_keys[:split_idx]
                    else:  # split == 'test' (validation split)
                        selected_keys = shuffled_keys[split_idx:]
                        
            self.image_pairs = [
                (ir_files[k], vis_files[k])
                for k in selected_keys
            ]
            
        # Define transform pipeline
        transform_list = []
        if self.resize_shape is not None:
            transform_list.append(transforms.Resize(self.resize_shape))
            
        if self.convert_to_grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
            
        transform_list.append(transforms.ToTensor())  # Scales automatically to [0.0, 1.0]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        ir_path, vis_path = self.image_pairs[idx]
        
        # Open in RGB to avoid issues with some images being grayscale and others colored
        ir_img = Image.open(ir_path).convert('RGB')
        vis_img = Image.open(vis_path).convert('RGB')
        
        # Apply transformation pipeline
        ir_tensor = self.transform(ir_img)
        vis_tensor = self.transform(vis_img)
        
        return {
            "ir": ir_tensor,
            "vis": vis_tensor,
            "ir_path": ir_path,
            "vis_path": vis_path
        }

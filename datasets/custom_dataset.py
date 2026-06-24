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
    def __init__(self, root_dir, resize_shape=(64, 64), convert_to_grayscale=True, is_train=True, val_split=0.2, seed=42):
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
            # Gather valid image files
            ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
            vis_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
            
            # Find common base names (keys) to form pairs
            common_names = sorted(list(set(ir_files.keys()).intersection(vis_files.keys())))
            
            # Alignment diagnostics
            missing_in_vis = ir_files.keys() - vis_files.keys()
            missing_in_ir = vis_files.keys() - ir_files.keys()
            
            if missing_in_vis:
                print(f"WARNING: {len(missing_in_vis)} infrared files are missing matching visible files.")
            if missing_in_ir:
                print(f"WARNING: {len(missing_in_ir)} visible files are missing matching infrared files.")
            
            print(f"Total matched pairs: {len(common_names)}")
            
            # Split into train/validation sets deterministically
            import random
            random.seed(seed)
            random.shuffle(common_names)
            
            split_idx = int(len(common_names) * (1 - val_split))
            if self.is_train:
                selected_names = common_names[:split_idx]
            else:
                selected_names = common_names[split_idx:]
                
            self.image_pairs = [
                (os.path.join(self.ir_dir, ir_files[name]), os.path.join(self.vis_dir, vis_files[name]))
                for name in selected_names
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

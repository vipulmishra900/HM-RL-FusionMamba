import argparse
import sys
import torch

from config import HMRLFusionMambaConfig
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from complexity_analysis.benchmark import run_complexity_benchmark

def parse_args():
    parser = argparse.ArgumentParser(description="HM-RL-FusionMamba Framework CLI")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "eval", "benchmark", "test-dataset"],
        help="Execution mode: train, eval (evaluate model), benchmark (complexity analysis), or test-dataset (verify dataloader)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to model checkpoint (required for 'eval' mode)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        choices=["cpu", "cuda"],
        help="Computation device (defaults to value in config.py)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def test_dataset_pipeline(config):
    """
    Test loading dataset, print tensor shapes, verify dataloader works,
    and validate the RL compatibility wrapper (RLDatasetAdapter).
    If no images are found, generates dummy images temporarily for testing.
    """
    import os
    import sys
    from torch.utils.data import DataLoader
    from datasets.custom_dataset import PairedImageDataset
    from datasets.rl_dataset_adapter import RLDatasetAdapter
    from verify_dataset import generate_dummy_images, delete_dummy_images

    base_path = os.path.dirname(os.path.abspath(__file__))
    llvip_dir = os.path.join(base_path, 'datasets', 'LLVIP')
    flir_dir = os.path.join(base_path, 'datasets', 'FLIR')

    # Read image size dynamically from config
    resize_shape = config.encoder.visual_input_shape[1:]
    print(f"Configured image size from config.py: {resize_shape}")

    # Count actual image pairs
    def count_pairs(root_dir):
        ir_dir = os.path.join(root_dir, 'infrared')
        vis_dir = os.path.join(root_dir, 'visible')
        if not os.path.exists(ir_dir) or not os.path.exists(vis_dir):
            return 0
        ir_files = {os.path.splitext(f)[0] for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        vis_files = {os.path.splitext(f)[0] for f in os.listdir(vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        return len(ir_files.intersection(vis_files))

    llvip_pairs = count_pairs(llvip_dir)
    flir_pairs = count_pairs(flir_dir)

    target_dir = llvip_dir
    temp_testing = False

    if llvip_pairs == 0 and flir_pairs == 0:
        print("No real images found for LLVIP or FLIR. Generating dummy images for main.py test...")
        generate_dummy_images(llvip_dir, num_pairs=8)
        temp_testing = True
    elif llvip_pairs > 0:
        print("Using LLVIP dataset for testing...")
        target_dir = llvip_dir
    else:
        print("Using FLIR dataset for testing...")
        target_dir = flir_dir

    try:
        # 1. Base Dataset Test
        dataset = PairedImageDataset(
            root_dir=target_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=True,
            val_split=0.2
        )
        print(f"Loaded dataset from {target_dir} successfully.")
        print(f"Dataset length (train split): {len(dataset)}")
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        print("Created DataLoader successfully.")
        
        batch = next(iter(dataloader))
        print("Successfully loaded a base dataset batch!")
        print(f"IR tensor shape: {batch['ir'].shape}")
        print(f"Visible tensor shape: {batch['vis'].shape}")
        print(f"IR tensor range: [{batch['ir'].min():.4f}, {batch['ir'].max():.4f}]")
        print(f"Visible tensor range: [{batch['vis'].min():.4f}, {batch['vis'].max():.4f}]")
        
        assert batch['ir'].shape[2:] == resize_shape, f"IR shape mismatch! Expected {resize_shape}, got {batch['ir'].shape[2:]}"
        assert batch['vis'].shape[2:] == resize_shape, f"Vis shape mismatch! Expected {resize_shape}, got {batch['vis'].shape[2:]}"
        assert batch['ir'].min() >= 0.0 and batch['ir'].max() <= 1.0, "IR normalization check failed!"
        assert batch['vis'].min() >= 0.0 and batch['vis'].max() <= 1.0, "Vis normalization check failed!"
        print("Base dataset assertions PASSED!")

        # 2. RL Dataset Adapter Test
        print("\nTesting RLDatasetAdapter wrapper...")
        rl_dataset = RLDatasetAdapter(dataset, config)
        rl_dataloader = DataLoader(rl_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        rl_batch = next(iter(rl_dataloader))
        print("Successfully loaded an RL adapter batch!")
        print(f"RL Batch keys: {list(rl_batch.keys())}")
        print(f"Visual observation tensor shape: {rl_batch['visual'].shape}")
        print(f"Proprioceptive observation tensor shape: {rl_batch['proprio'].shape}")
        print(f"Action tensor shape: {rl_batch['action'].shape}")
        print(f"Reward tensor shape: {rl_batch['reward'].shape}")
        
        # Verify RL shapes
        expected_channels = (1 if dataset.convert_to_grayscale else 3) * 2  # concat vis and ir
        assert rl_batch['visual'].shape[1] == expected_channels, f"RL visual channels mismatch! Expected {expected_channels}, got {rl_batch['visual'].shape[1]}"
        assert rl_batch['visual'].shape[2:] == resize_shape, f"RL visual spatial mismatch! Expected {resize_shape}, got {rl_batch['visual'].shape[2:]}"
        assert rl_batch['proprio'].shape[1] == config.encoder.proprioceptive_dim, f"RL proprio dim mismatch!"
        assert rl_batch['action'].shape[1] == config.agent.action_dim, "RL action dim mismatch!"
        print("RL Adapter assertions PASSED!")

    except Exception as e:
        print(f"Dataset test failed with error: {e}", file=sys.stderr)
        if temp_testing:
            delete_dummy_images(llvip_dir, num_pairs=8)
        sys.exit(1)

    if temp_testing:
        delete_dummy_images(llvip_dir, num_pairs=8)

def main():
    args = parse_args()
    
    # Load configuration
    config = HMRLFusionMambaConfig()
    
    # Override configuration with CLI arguments
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed
        
    # Check device availability
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        config.training.device = "cpu"
        
    print("=" * 60)
    print(f"HM-RL-FusionMamba Framework running in mode: {args.mode.upper()}")
    print(f"Device: {config.training.device.upper()}")
    print("=" * 60)
    
    if args.mode == "train":
        trainer = Trainer(config)
        trainer.train()
        
    elif args.mode == "eval":
        if args.checkpoint is None:
            print("ERROR: --checkpoint must be provided in 'eval' mode.", file=sys.stderr)
            sys.exit(1)
        evaluator = Evaluator(config, checkpoint_path=args.checkpoint)
        evaluator.evaluate()
        
    elif args.mode == "benchmark":
        print("Running complexity and performance analysis...")
        run_complexity_benchmark(config)
        
    elif args.mode == "test-dataset":
        test_dataset_pipeline(config)
        
if __name__ == "__main__":
    main()

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
        choices=["train", "eval", "benchmark"],
        help="Execution mode: train, eval (evaluate model), or benchmark (complexity analysis)"
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
        
if __name__ == "__main__":
    main()

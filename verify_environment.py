import sys

def main():
    print("=" * 60)
    print("        HM-RL-FusionMamba Environment Verification")
    print("=" * 60)
    
    # 1. Verify Python Version
    print(f"Python version: {sys.version.split()[0]}")
    
    # 2. Verify Packages
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("cv2", "opencv"),
        ("skimage", "scikit-image"),
        ("stable_baselines3", "stable-baselines3"),
        ("gymnasium", "gymnasium"),
        ("einops", "einops"),
        ("timm", "timm"),
        ("fvcore", "fvcore"),
        ("thop", "thop"),
        ("tensorboard", "TensorBoard")
    ]
    
    print("\nVerifying package imports and versions:")
    all_passed = True
    for module_name, name in packages:
        try:
            mod = __import__(module_name)
            # Fetch version attributes
            if module_name == "cv2":
                version = getattr(mod, "__version__", "Installed")
            elif module_name == "tensorboard":
                version = getattr(mod, "__version__", "Installed")
            else:
                version = getattr(mod, "__version__", "Installed")
            print(f"  - {name:<20}: Version {version}")
        except ImportError as e:
            print(f"  - {name:<20}: NOT INSTALLED ({e})")
            all_passed = False

    # 3. Verify CUDA Availability
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
        
    print(f"\nCUDA Available: {cuda_available}")
    
    # 4. Verify CPU Execution Status
    cpu_executable = True # Python and local processes run on CPU
    print(f"CPU Execution Status: Enabled (Available for computation)")

    print("=" * 60)
    if all_passed:
        print("Verification status: SUCCESS")
    else:
        print("Verification status: FAILED (Some packages are missing)")
    print("=" * 60)

if __name__ == "__main__":
    main()

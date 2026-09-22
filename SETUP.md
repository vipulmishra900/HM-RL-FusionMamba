# Setup Instructions: HM-RL-FusionMamba Environment

This guide describes how to configure, activate, and verify the Python 3.10.11 environment for **HM-RL-FusionMamba**.

---

## 1. Dependencies

The dependencies are divided into two modular categories to allow simple transitions between CPU and GPU hardware:

### A. PyTorch Core (Installed via Setup Scripts)
- **PyTorch** & **Torchvision**: Target versions are modularized and installed dynamically based on your available hardware (CPU vs. NVIDIA GPU).

### B. Core Scientific & RL Libraries (Installed via requirements.txt)
- **stable-baselines3** & **gymnasium**: Reinforcement learning baseline models and environments.
- **timm** & **einops**: Vision network backbones and tensor rearrangement.
- **numpy**, **scipy**, **opencv-python**, **scikit-image**, **matplotlib**: Numerical modeling, scientific calculation, vision filters, and plotting.
- **fvcore** & **thop**: Complexity profiling (parameter and FLOPs analysis).
- **pyyaml** & **tensorboard**: Configuration loading and tracking logs.

---

## 2. Installation Instructions

We provide two automated PowerShell scripts to handle the environment creation, modular library resolution, and validation.

### Prerequisites
Make sure you have [Conda (Anaconda/Miniconda)](https://docs.conda.io/en/latest/miniconda.html) installed and configured on your PowerShell terminal.

### Option A: Install for CPU Execution (Default/Current)
To build a CPU-compatible development environment, run the CPU setup script in PowerShell:
```powershell
./setup_cpu.ps1
```
This script will:
1. Create the `hm-rl-fusionmamba` Conda environment using [environment.yml](environment.yml) (Python 3.10.11).
2. Install the PyTorch CPU-only distribution wheel.
3. Install all core libraries listed in [requirements.txt](requirements.txt).
4. Run validation checks.

### Option B: Install for GPU Execution (NVIDIA / CUDA Support)
To build a GPU-accelerated environment with CUDA support, run the GPU setup script in PowerShell:
```powershell
./setup_gpu.ps1
```
This script will:
1. Create the `hm-rl-fusionmamba` Conda environment using [environment.yml](environment.yml) (Python 3.10.11).
2. Install PyTorch with CUDA 12.1 GPU support.
3. Install all core libraries listed in [requirements.txt](requirements.txt).
4. Run validation checks.

---

## 3. Environment Activation

Once setup is complete, activate your Conda environment using:

```bash
conda activate hm-rl-fusionmamba
```

---

## 4. Verification

To manually re-verify the environment state and hardware capabilities at any time, run:

```bash
python verify_environment.py
```

This runs:
- An import integrity check on all 14 dependency modules.
- A CUDA acceleration detection reporting your GPU device specs and active CUDA driver version.
- A 4000x4000 matrix multiplication latency profiling speed test.

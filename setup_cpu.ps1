# ==============================================================================
# PowerShell Setup Script - HM-RL-FusionMamba (CPU Version)
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         Setting up HM-RL-FusionMamba CPU Environment" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Conda Installation
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Conda is not recognized. Please install Anaconda or Miniconda and add it to your PATH." -ForegroundColor Red
    Exit 1
}

# 2. Build Conda base env (Python 3.10.11)
Write-Host "`n[1/4] Building Conda environment 'hm-rl-fusionmamba' with Python 3.10.11..." -ForegroundColor Yellow
conda env create -f environment.yml --force

# 3. Modularly install PyTorch CPU version
Write-Host "`n[2/4] Installing modular PyTorch (CPU-only build)..." -ForegroundColor Yellow
conda run -n hm-rl-fusionmamba pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining core dependencies
Write-Host "`n[3/4] Installing core dependencies from requirements.txt..." -ForegroundColor Yellow
conda run -n hm-rl-fusionmamba pip install -r requirements.txt

# 5. Run Verification Script
Write-Host "`n[4/4] Verifying installation..." -ForegroundColor Yellow
conda run -n hm-rl-fusionmamba python verify_environment.py

Write-Host "`nSetup Completed! Activate the environment using: conda activate hm-rl-fusionmamba" -ForegroundColor Green

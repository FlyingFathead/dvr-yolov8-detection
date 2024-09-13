#!/bin/bash

# Name of the conda environment
ENV_NAME="yolov8_env"

# Desired Python version
PYTHON_VERSION="3.10"

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed."
    echo "Please install Miniconda or Anaconda and try again."
    exit 1
fi

# Create the conda environment with the specified Python version
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda install -y mamba
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
mamba activate $ENV_NAME

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust the CUDA version as needed)
echo "Installing PyTorch with CUDA support..."
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge ffmpeg

# Install the required packages
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Deactivate the conda environment after setup
echo "Deactivating conda environment."
mamba deactivate

echo "Setup complete. To activate the conda environment, run:"
echo "conda activate $ENV_NAME"

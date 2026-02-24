#!/usr/bin/env bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    software-properties-common

# Install audio libraries (critical for librosa/soundfile)
sudo apt install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-extra \
    portaudio19-dev \
    libffi-dev

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Check if NVIDIA driver is installed
nvidia-smi

# If not installed, install the recommended driver
sudo apt install -y nvidia-driver-545

# Reboot required after driver install
sudo reboot

# After reboot, verify
nvidia-smi

# Download and install CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version

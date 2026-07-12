# GLM-5.2-NVFP4 Deployment Setup Guide

Based on your command history, here's a structured, repeatable setup script. I've removed exploratory/repeated commands and organized it into logical phases.

## 📋 Prerequisites
- Ubuntu 24.04 (ARM64) with NVIDIA GPUs (4x for TP=4)
- NVMe disk at `/dev/nvme1n1` (adjust as needed)
- Sudo access
- HuggingFace token with access to `nvidia/GLM-5.2-NVFP4`

---

## 🚀 Setup Script

```bash
#!/bin/bash
set -euo pipefail

# ============================================================
# CONFIGURATION - Edit these as needed
# ============================================================
HF_TOKEN="hf_YOUR_TOKEN_HERE"                    # <-- Replace with your token
MODEL_ID="nvidia/GLM-5.2-NVFP4"
MODEL_DIR="/mnt/scratch/huggingface/GLM-5.2-NVFP4"
NVME_DEVICE="/dev/nvme1n1"
MOUNT_POINT="/mnt/scratch"
HF_VENV="$HOME/hf-env"
GCS_BUCKET="your-gcs-bucket-name"                # <-- Replace or skip GCS step
GCS_LOCATION="us-east4"
SGLANG_PORT=8002

# ============================================================
# PHASE 1: SYSTEM INSPECTION (optional, informational)
# ============================================================
echo "=== Phase 1: System Inspection ==="
hostname -f
df -hk
lsblk
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink -s | head
ibv_devices
ibv_devinfo | grep -E "hca_id|state|link_layer" || true

# ============================================================
# PHASE 2: STORAGE SETUP - Mount NVMe scratch disk
# ============================================================
echo "=== Phase 2: Mounting NVMe scratch disk ==="

# Format the NVMe disk (WARNING: destroys existing data!)
sudo mkfs.ext4 -F ${NVME_DEVICE}

# Create mount point and mount
sudo mkdir -p ${MOUNT_POINT}
sudo mount -o discard,defaults,nobarrier ${NVME_DEVICE} ${MOUNT_POINT}

# Make writable by all users
sudo chmod a+w ${MOUNT_POINT}

# Create model directory
mkdir -p ${MOUNT_POINT}/huggingface

df -hk

# ============================================================
# PHASE 3: HUGGINGFACE ENVIRONMENT SETUP
# ============================================================
echo "=== Phase 3: Setting up HuggingFace environment ==="

# Create Python venv for HF tools
python3 -m venv ${HF_VENV}
source ${HF_VENV}/bin/activate

# Install/upgrade huggingface_hub with transfer acceleration
pip install -U "huggingface_hub[hf_transfer]"

# Auto-activate venv on future logins
echo "source ${HF_VENV}/bin/activate" >> ~/.bashrc

# Set environment variables
export HF_TOKEN=${HF_TOKEN}
export HF_HUB_ENABLE_HF_TRANSFER=1
# Note: XET disabled per your history (commands 51-52)
export HF_HUB_DISABLE_XET=1

# ============================================================
# PHASE 4: DOWNLOAD MODEL
# ============================================================
echo "=== Phase 4: Downloading ${MODEL_ID} ==="

# Download model with optimal worker count
hf download ${MODEL_ID} \
    --local-dir ${MODEL_DIR}

# Verify download
ls -ltr ${MODEL_DIR}

# ============================================================
# PHASE 5: GCS UPLOAD (optional - for checkpoint backup)
# ============================================================
echo "=== Phase 5: GCS upload (optional) ==="
# Only run if gsutil is available
if command -v gsutil &> /dev/null; then
    gsutil mb -l ${GCS_LOCATION} gs://${GCS_BUCKET}/ || true
    cd ${MODEL_DIR}
    gsutil -m cp -r * gs://${GCS_BUCKET}/nvfp4/
    cd ~
else
    echo "gsutil not installed - skipping GCS upload"
fi

# ============================================================
# PHASE 6: DOCKER INSTALLATION
# ============================================================
echo "=== Phase 6: Installing Docker ==="

# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository (ARM64)
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list

# Install Docker
sudo apt-get update
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# ============================================================
# PHASE 7: NVIDIA CONTAINER TOOLKIT
# ============================================================
echo "=== Phase 7: Installing NVIDIA Container Toolkit ==="

# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA container toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# ============================================================
# PHASE 8: DEEPEP SETUP (optional - for expert parallelism)
# ============================================================
echo "=== Phase 8: DeepEP setup ==="

# Install PyTorch with CUDA 13.0
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Clone DeepEP
cd ~
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP

# Set CUDA arch for your GPU (10.3 for GB300)
export TORCH_CUDA_ARCH_LIST="10.3"

# Install build dependencies
pip install ninja

# Install NCCL
pip install -U "nvidia-nccl-cu13>=2.30.4"

# Set NCCL paths for build
export NCCL_ROOT=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])")
export EP_NCCL_ROOT_DIR=$NCCL_ROOT
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

# Build DeepEP
rm -rf build/
pip install --no-build-isolation .

# Verify installation
python -c "import deep_ep; print(deep_ep.__version__)"

# Run intranode test
python tests/test_intranode.py

# ============================================================
# PHASE 9: LAUNCH SGLANG SERVER
# ============================================================
echo "=== Phase 9: Launching SGLang server ==="

# Stop any existing container
docker rm -f glm52_sglang_gb300_server 2>/dev/null || true

# Launch SGLang server in detached mode
sudo docker run -d \
    --name glm52_sglang_gb300_server \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    -p ${SGLANG_PORT}:${SGLANG_PORT} \
    -e SGLANG_OPT_USE_TOPK_V2=1 \
    -v ${MODEL_DIR}:/model \
    lmsysorg/sglang:dev-cu13 \
    python3 -m sglang.launch_server \
        --model-path /model \
        --tensor-parallel-size 4 \
        --quantization modelopt_fp4 \
        --max-running-requests 16 \
        --max-prefill-tokens 8192 \
        --chunked-prefill-size 8192 \
        --cuda-graph-max-bs 16 \
        --mem-fraction-static 0.87 \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --bf16-gemm-backend cutedsl \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 5 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 6 \
        --host 0.0.0.0 \
        --port ${SGLANG_PORT}

# Show container status
docker ps

# Tail logs (Ctrl+C to exit without stopping container)
echo "=== Tailing server logs (Ctrl+C to exit) ==="
docker logs -f glm52_sglang_gb300_server

# ============================================================
# VERIFICATION
# ============================================================
# Once server is up, test with:
# curl http://localhost:8002/v1/models
# curl http://localhost:8002/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 📝 Key Notes

| Item | Value |
|------|-------|
| **Model** | `nvidia/GLM-5.2-NVFP4` |
| **Quantization** | `modelopt_fp4` |
| **Tensor Parallel** | 4 GPUs |
| **Server Port** | 8002 |
| **Container Name** | `glm52_sglang_gb300_server` |
| **Docker Image** | `lmsysorg/sglang:dev-cu13` |

## ⚠️ Important Notes

1. **NVMe Device**: Verify `/dev/nvme1n1` exists with `lsblk` before running — adjust if different
2. **HF Token**: Replace the placeholder with your actual HuggingFace token
3. **XET Disabled**: Your history shows XET was disabled (commands 51-52) — kept that setting
4. **Context Length**: Removed `--context-length 90000` from final run (per command 100)
5. **GCS Upload**: Optional step — skip if not using Google Cloud Storage
6. **DeepEP**: Optional — only needed for expert parallelism testing
7. **CUDA Arch**: `10.3` is for GB300 — adjust for your GPU architecture
8. **Docker Group**: Run `newgrp docker` or log out/in after adding user to docker group

## 🔧 Useful Management Commands

```bash
# Check server status
docker ps
docker logs -f glm52_sglang_gb300_server

# Stop server
docker stop glm52_sglang_gb300_server
docker rm glm52_sglang_gb300_server

# GPU monitoring
nvidia-smi
watch -n 1 nvidia-smi

# If GPUs get stuck
sudo nvidia-smi --gpu-reset
sudo systemctl restart docker
```

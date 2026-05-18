# Kimi-K2.6 Variant B — 2-node 2× SMG-routed replicas + EAGLE3 (4, 6) + HiCache.
#
# Each node runs ONE independent single-node TP=8 sglang server with the
# launch command below. A single SMG router (on either node) routes traffic
# to both replicas.

# ============================================================
# Per-server environment (exported on each node before launch)
# ============================================================
FLASHINFER_DISABLE_VERSION_CHECK=1 \
SGLANG_ENABLE_SPEC_V2=1 \
NCCL_DEBUG=WARN \
NCCL_SOCKET_IFNAME=enp128s4,ens4 \
GLOO_SOCKET_IFNAME=ens4 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_SOCKET_NTHREADS=8 \
NCCL_NSOCKS_PERTHREAD=8 \
NCCL_MIN_NCHANNELS=8 \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
NCCL_NVLS_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
OMP_NUM_THREADS=24 \
SGLANG_SET_CPU_AFFINITY=1 \
SAFETENSORS_FAST_GPU=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_ENABLE_DEEP_GEMM=0 \
SGLANG_ENABLE_JIT_DEEPGEMM=0

# ============================================================
# Image: lmsysorg/sglang:dev-cu13
# Inside the container, before launch:
#   pip install sglang==0.5.10.post1
# ============================================================

# ============================================================
# Per-server launch — run identically on each of the two nodes.
# Listens on :30000 (the address the router will target).
# ============================================================
python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3 \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through_selective \
  --schedule-policy lpm \
  --schedule-conservativeness 0.3 \
  --chunked-prefill-size 16384 \
  --enable-mixed-chunk \
  --context-length 131072 \
  --max-running-requests 128 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --enable-p2p-check \
  --host 0.0.0.0 \
  --port 30000

# ============================================================
# SMG router — run once (on either node), points at both replicas.
# Exposes a single OpenAI-compatible endpoint on :8000.
# ============================================================
smg launch \
  --worker-urls http://<NODE1_PRIVATE_IP>:30000 http://<NODE2_PRIVATE_IP>:30000 \
  --policy power_of_two \
  --host 0.0.0.0 \
  --port 8000 \
  --request-timeout-secs 1800 \
  --max-concurrent-requests 512

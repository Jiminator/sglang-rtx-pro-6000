#!/bin/bash
# GLM-5.2-NVFP4 + DP-attention — 1K/8K WINNER on 1x g4 (8x RTX PRO 6000, SM120).
# nvidia/GLM-5.2-NVFP4 (GlmMoeDsa DSA arch, expert-only NVFP4), STOCK latest-main dev-cu13, bench_serving rrr=1.0.
# STEADY-STATE DECODE PLATEAU: ~3379 output tok/s (~422/GPU) @ 64 running-req/rank, gsm8k 0.900.
# Re-verified 2026-08-11 on sglang 0.0.0.dev1+gd59c1ddf7. See TUNING_REPORT.md.
#
# ############################################################################
# ## THREE FIXES vs the 2026-07-05 version of this script. Latest main needs ##
# ## ALL THREE or the server dies at launch / dies at init / SERVES GARBAGE. ##
# ############################################################################
#
# (1) --cuda-graph-bs values MUST BE UNQUOTED (this script used to quote them).
#     "16 32 48 64" reaches argparse as ONE token and hard-fails:
#       sglang serve: error: argument --cuda-graph-bs: invalid int value: '16 32 48 64'
#     On Kubernetes/GKE each int must be its OWN element of the args: list:
#       - --cuda-graph-bs
#       - "16"
#       - "32"
#       - "48"
#       - "64"
#     A container dying on an argparse error just looks like CrashLoopBackOff.
#
# (2) --disable-piecewise-cuda-graph is now REQUIRED.
#     The prefill-CUDA-graph capture added since 07-05 unconditionally dereferences the DSA
#     indexer metadata, which is None under --attention-backend flashinfer on this stack:
#       dsa_prefill_cuda_graph.py:122 -> dsa_indexer.py:1225 metadata.get_seqlens_expanded()
#       AttributeError: 'NoneType' object has no attribute 'get_seqlens_expanded'
#       RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)
#     All 8 schedulers die. (Flag did not exist on the older b28bc1060 pin — drop it there.)
#
# (3) --dsa-{prefill,decode}-backend trtllm is now REQUIRED. ***SILENT CORRUPTION*** otherwise.
#     Latest main auto-selects a new path for this hw+dtype:
#       "Set DSA backends for GLM FP8 KV Cache on SM120/SM121: prefill=flashinfer_sparse_mla,
#        decode=flashinfer_sparse_mla"
#     It BOOTS CLEAN AND SERVES AT FULL SPEED but the output is wrong: gsm8k 0.000 / Invalid
#     1.000; completions emit a correct FIRST token then collapse to repeated '!' (token 0) =
#     NaN logits in decode. Nothing in the logs flags it. Forcing trtllm (the 07-05 path)
#     restores gsm8k 0.900 AND enlarges the KV pool 205,184 -> 228,352 tok/rank.
#     ==> ALWAYS run the gsm8k gate below before believing any throughput number. <==
#
# HEADLINE (unchanged): fp8_e4m3 KV DSA decode works on STOCK latest main. The ONLY env var needed
# for fp8 is SGLANG_DISABLE_DSA_INDEXER_FUSION=1 (ablation A1a: gsm8k 0.940). The DSV4 vars
# (SGLANG_SM120_FLASHMLA_BACKEND / SGLANG_OPT_FLASHMLA_SPARSE_PREFILL) are INERT for GLM (different
# DSA dispatch). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is for boot.
# NVFP4 MoE runner = flashinfer_cutlass (only viable on SM120; marlin gsm8k 0.02). Pool is the lever;
# mfs0.975 ceiling. --cuda-graph-bs is a MEMORY lever, not just a batch cap: dropping it OOMs at capture.
set -euo pipefail
NAME=${NAME:-glm52_nvfp4_dpa_1k8k}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-cu13}
# nvidia/GLM-5.2-NVFP4. The old b0b2b68 pin is STALE — HF refs/main is now aec724e8 (47 shards, 465 GB),
# which is what the 2026-08-11 re-run measured. Pin an explicit snapshot path under HF_HUB_OFFLINE if you
# need reproducibility.
CKPT=${CKPT:-nvidia/GLM-5.2-NVFP4}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_DISABLE_DSA_INDEXER_FUSION=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$CKPT" --trust-remote-code \
    --tp 8 --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend none --ep-size 1 --moe-runner-backend flashinfer_cutlass \
    --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
    --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
    --disable-shared-experts-fusion --disable-piecewise-cuda-graph \
    --mem-fraction-static 0.975 --chunked-prefill-size 2048 \
    --cuda-graph-bs 16 32 48 64 --context-length 9472 --max-running-requests 1024 \
    --host 0.0.0.0 --port 8000
# ---- CORRECTNESS GATE — run this BEFORE any throughput measurement (see fix 3) ----
#   python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --port 8000
#   Expect Accuracy ~0.900, Invalid 0.000. If Invalid is 1.000 the DSA override did not take effect.
#   Sanity-check /get_server_info shows dsa_prefill_backend=trtllm, dsa_decode_backend=trtllm.
#
# Bench (loadgen, read steady-state decode plateau from server logs):
#   python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
#     --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 1.0 \
#     --max-concurrency 512 --num-prompts 2048
#
# ---- ALTERNATES ----
# (a) EAGLE-3 spec variant. ⚠️ NOT re-verified on latest main, and EAGLE spec decode on SM120 is
#     reported BLOCKED by upstream regression #29787 (IndexShare) from v0.5.15 — see ../../README.md.
#     Historical (b28bc1060): ~330/GPU, gsm8k 0.940, accept-len 4.0, 14-16 running/rank, mfs 0.97. Add:
#       --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
#       --speculative-eagle-topk 1 --speculative-moe-runner-backend flashinfer_cutlass \
#       --speculative-moe-a2a-backend none
#     (--speculative-eagle-topk MUST be 1: flashinfer-MLA topk=1 only for spec on SM120.)
# (b) bf16 conservative fallback (~158/GPU on b28bc1060, gsm8k 0.920, pool-bound & decays): drop
#     SGLANG_DISABLE_DSA_INDEXER_FUSION, set --kv-cache-dtype bfloat16 --mem-fraction-static 0.94.
# (c) Do NOT drop --cuda-graph-bs. Tested 2026-08-11: at mfs 0.975 the default bucket list OOMs during
#     capture (686 MiB in cuda-graph private pools, 259 MiB free). Breakable cuda graph being the
#     default does not retire this flag.

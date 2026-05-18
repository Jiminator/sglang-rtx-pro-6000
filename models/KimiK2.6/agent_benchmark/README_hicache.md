# How to Build and Deploy sglang-kimi-hicach-pod

This guide outlines the steps to deploy the `sglang-kimi-hicach-pod` on GKE, which runs the SGLang server with Hierarchical Cache (HiCache) enabled and applies a custom patch to `radix_cache.py`.

## Prerequisites

1. **GKE Cluster**: A cluster with a node pool named `g4-384-np-0` equipped with NVIDIA GPUs.
2. **GCS Bucket**: A bucket named `shivaji-sglang-kimi-us-central1` containing the model weights in the folder `kimi-k2.6`.
3. **Hugging Face Secret**: A Kubernetes secret named `hf-secret` containing your Hugging Face token (`HF_TOKEN`).
   ```bash
   kubectl create secret generic hf-secret --from-literal=HF_TOKEN=your_token_here
   ```

## Step 1: Create the Radix Cache Patch ConfigMap

The pod mounts a custom version of `radix_cache.py` to override the one in the base image. You need to create a ConfigMap named `radix-cache-patch` from your local modified file.

If you have the modified file locally (e.g., `radix_cache_container.py` or `radix_cache.py`), run:

```bash
kubectl create configmap radix-cache-patch \
  --from-file=radix_cache.py=/path/to/your/local/radix_cache.py \
  -o yaml --dry-run=client | kubectl apply -f -
```

*Note: Ensure the key in `--from-file` is exactly `radix_cache.py` as specified in the pod volume mount `subPath`.*

## Step 2: Deploy the Pod

The deployment manifest is located at `/Users/shivajid/code/sglang/kimi/sglang-debug-hicache.yaml`. It contains both the `ServiceAccount` and the `Pod` definition.

To deploy, run:

```bash
kubectl apply -f /Users/shivajid/code/sglang/kimi/sglang-debug-hicache.yaml
```

## Step 3: Verify the Deployment

1. Check the pod status:
   ```bash
   kubectl get pod sglang-kimi-hicach-pod
   ```
2. View the logs to ensure the server starts correctly and HiCache is enabled:
   ```bash
   kubectl logs -f sglang-kimi-hicach-pod -c sglang-container
   ```

You should see SGLang server starting messages and logs indicating that `Hierarchical Cache` is enabled.

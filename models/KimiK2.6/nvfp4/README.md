# Kimi-K2.6 NVFP4 — 2-Node Profiling Setup

Deploys `nvidia/Kimi-K2.6-NVFP4` across 2 nodes (TP 8 × PP 2) under `nsys profile`, with profile reports written directly to GCS via the GCS Fuse CSI driver.

## How profile output reaches GCS

The pod mounts the bucket `shivaji-sglang-kimi-us-central1` (prefix `nsys-profiles/kimi-k2.6-nvfp4/`) at `/gcs-profiles` using GCS Fuse, and `nsys` writes its report there:

- Node 0 → `gs://shivaji-sglang-kimi-us-central1/nsys-profiles/kimi-k2.6-nvfp4/kimi26-nvfp4-node0.nsys-rep`
- Node 1 → `gs://shivaji-sglang-kimi-us-central1/nsys-profiles/kimi-k2.6-nvfp4/kimi26-nvfp4-node1.nsys-rep`

GCS Fuse stages writes locally and uploads when nsys closes the report file (after profiling stops), so the report lands in the bucket without any `kubectl cp`.

## Prerequisites (one-time)

1. The cluster must have Workload Identity and the GCS Fuse CSI driver enabled (already true for this cluster — see [gke-inference-gateway/sglang/README.md](../gke-inference-gateway/sglang/README.md)).

2. The `sglang-sa` Kubernetes service account must exist in the `default` namespace (already used by the 1-node config).

3. Grant **write** access on the bucket. The existing grants are read-only (`objectViewer`); writing profiles needs `objectUser`:

   ```bash
   gcloud storage buckets add-iam-policy-binding gs://shivaji-sglang-kimi-us-central1 \
     --member="principal://iam.googleapis.com/projects/9452062936/locations/global/workloadIdentityPools/northam-ce-mlai-tpu.svc.id.goog/subject/ns/default/sa/sglang-sa" \
     --role=roles/storage.objectUser
   ```

   If the cluster uses the legacy Workload Identity binding style (as in the other READMEs):

   ```bash
   gcloud storage buckets add-iam-policy-binding gs://shivaji-sglang-kimi-us-central1 \
     --member="serviceAccount:northam-ce-mlai-tpu.svc.id.goog[default/sglang-sa]" \
     --role=roles/storage.objectUser
   ```

## Deploy

```bash
kubectl apply -f sglang-kimi26-nvfp4-2node.yaml
```

Wait for both pods to be ready and the server to finish loading:

```bash
kubectl get pods -l app=distributed-sglang-nvfp4-k26 -w
kubectl logs distributed-sglang-nvfp4-k26-0 -f
```

## Capture a profile

nsys starts dormant (`--capture-range=cudaProfilerApi`) — tracing only runs between the start/stop calls below.

```bash
# Port-forward or use the NodePort of sglang-serving-nvfp4-k26
kubectl port-forward svc/sglang-serving-nvfp4-k26 8000:8000 &

# 1. Start capture
curl -X POST http://localhost:8000/start_profile

# 2. Send the workload you want to profile (bench_serving, real traffic, etc.)

# 3. Stop capture — nsys finalizes the report and GCS Fuse uploads it
curl -X POST http://localhost:8000/stop_profile
```

## Retrieve the reports

```bash
gcloud storage ls gs://shivaji-sglang-kimi-us-central1/nsys-profiles/kimi-k2.6-nvfp4/
gcloud storage cp "gs://shivaji-sglang-kimi-us-central1/nsys-profiles/kimi-k2.6-nvfp4/*.nsys-rep" .
```

Open the `.nsys-rep` files locally in Nsight Systems.

## Notes

- Reports can be large (multi-GB at 16-GPU scale); the gcsfuse sidecar is given 50Gi ephemeral storage for write staging. Bump `gke-gcsfuse/ephemeral-storage-limit` in the pod annotations if uploads fail on long captures.
- Throughput numbers from this deployment are not clean benchmark numbers — the nsys wrapper adds overhead even when dormant. Use a config without the nsys wrapper for benchmarking.

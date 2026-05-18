# K8s Workload & SGLang Metrics Monitor

A lightweight web application to monitor Kubernetes workloads in your cluster and visualize SGLang server metrics, including KV cache usage, HiCache status, and benchmark progress.

## Features
- **Workload Status**: View running pods grouped by GKE nodepool.
- **Log Streaming**: Click on a pod name to view streaming logs (SSE).
- **Metrics Visualization**: View real-time metrics for selected SGLang pods (KV cache, Hit Rate, Throughput, CPU, Memory, and HiCache).
- **NVIDIA SMI**: Execute `nvidia-smi` on target pods directly from the UI.
- **Benchmark Progress**: Track active sequences and overall step completion for running benchmark jobs.
- **Benchmark Summary**: View full JSON summaries for completed benchmark runs.

## Project Structure
Located in `/Users/shivajid/code/sglang/kimi/k8s-monitor/`:
- `server.py`: Python HTTP server handling API requests and proxying to K8s API and SGLang metrics endpoints.
- `index.html`: Single-page application frontend with premium dark-mode UI and glassmorphism aesthetics.
- `k8s-manifests.yaml`: Kubernetes manifests for Deployment, Service, ServiceAccount, and RBAC roles.
- `sequence_steps.json`: Static mapping of total steps per sequence (copied from benchmark data) used for granular progress tracking.

## Deployment Steps

To avoid the need for building and pushing custom Docker images, this application is deployed by mounting the code directly into a standard `python:3.9-slim` image via a Kubernetes ConfigMap.

### Step 1: Create/Update the ConfigMap
Run this command to create the ConfigMap from the local source files. This allows you to update the UI or server logic and apply changes without rebuilding images:

```bash
kubectl create configmap k8s-monitor-code \
  --from-file=/Users/shivajid/code/sglang/kimi/k8s-monitor/server.py \
  --from-file=/Users/shivajid/code/sglang/kimi/k8s-monitor/index.html \
  --from-file=/Users/shivajid/code/sglang/kimi/k8s-monitor/sequence_steps.json \
  -o yaml --dry-run=client | kubectl apply -f -
```

### Step 2: Apply Kubernetes Manifests
Apply the manifests to create the ServiceAccount, Roles, Deployment, and Service:

```bash
kubectl apply -f /Users/shivajid/code/sglang/kimi/k8s-monitor/k8s-manifests.yaml
```

### Step 3: Access the UI
Since the service is created as a `ClusterIP` (internal to the cluster), you need to use port-forwarding to access it from your Mac:

```bash
kubectl port-forward svc/k8s-monitor 8080:80
```

Now you can open your browser and navigate to [http://localhost:8080](http://localhost:8080).

## Updating the Application
Whenever you make changes to `server.py` or `index.html`:
1. Re-run **Step 1** to update the ConfigMap.
2. Restart the deployment to pick up the changes:
   ```bash
   kubectl rollout restart deployment/k8s-monitor
   ```

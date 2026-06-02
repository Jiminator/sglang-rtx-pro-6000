# GKE Inference Gateway: Prefix-Cache Aware Routing

This repository contains the configuration files and test scripts to set up and verify **Precise Prefix-Cache Aware Routing** using GKE Inference Gateway and SGLang.

This setup ensures that requests sharing the same prompt prefix are routed to the same SGLang replica, maximizing KV cache reuse and reducing Time-to-First-Token (TTFT).

## Folder Structure

*   `inference-gateway/`:
    *   `epp-values.yaml`: Helm values file for deploying the Endpoint Picker (EPP).
    *   `gateway.yaml`: Kubernetes Gateway resource (using `gke-l7-rilb` for internal load balancing).
    *   `httproute.yaml`: Kubernetes HTTPRoute resource to bind traffic to the InferencePool.
    *   `inference-objective.yaml`: InferenceObjective resource for routing priority.
    *   `vllm-chat-app.yaml`: Deployment file for a lightweight Python CLI chat client.
    *   `tests/`:
        *   `prefix_cache_benchmark.py`: Functional test for prefix caching (sequential).
        *   `prefix_cache_concurrency_test.py`: Test with 20 concurrent requests of the same prefix.
        *   `prefix_specific_test.py`: Test with two different prefixes to verify targeted routing.
        *   `prefix_unbalanced_test.py`: Test with unbalanced load (10 vs 20 requests) for two prefixes.
        *   `high_concurrency_test.py`: Test with 50 concurrent requests.
        *   `extreme_concurrency_test.py`: Test with 100 concurrent requests.
        *   `massive_stress_test.py`: Test with 200 concurrent requests.
*   `kimi-k2.6/`: SGLang deployment manifests for KimiK2.6 model servers.
    *   `sglang-kimik2.6-1node.yaml`: Pod manifest for SGLang replica.
    *   `sglang-kimi-podmonitoring.yaml`: PodMonitoring resource for metrics.

---

## Prerequisites

1.  **GKE Cluster**: Version 1.34+ recommended (for native InferencePool support).
2.  **HuggingFace Secret**: A Kubernetes secret named `hf-secret` containing your HuggingFace token:
    ```bash
    kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<your_token>
    ```
3.  **Model Weights**: A GCS bucket containing the KimiK2.6 model weights, referenced in `kimi-k2.6/sglang-kimik2.6-1node.yaml` (configured as `shivaji-sglang-kimi-us-central1/kimi-k2.6`).
4.  **InferenceExtension CRDs**: If your cluster version does not have them installed by default, install them manually:
    ```bash
    IGW_LATEST_RELEASE=$(curl -s https://api.github.com/repos/kubernetes-sigs/gateway-api-inference-extension/releases \
      | grep tag_name | head -1 | cut -d '"' -f 4)

    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/${IGW_LATEST_RELEASE}/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml
    ```

---

## Step-by-Step Setup

### 1. Enable Gateway API on GKE
Ensure your cluster has the Gateway API enabled and HTTP Load Balancing active:

```bash
gcloud container clusters update shivaji-minimax-g4-384-cluster \
  --gateway-api=standard \
  --zone=us-central1-f \
  --project=northam-ce-mlai-tpu
```

### 2. Create Proxy-Only Subnet (For Internal Load Balancing)
If you are using internal load balancing (as configured in `inference-gateway/gateway.yaml` with `gke-l7-rilb`), you must create a proxy-only subnet in your VPC and region:

```bash
gcloud compute networks subnets create shivaji-proxy-only-subnet \
  --purpose=REGIONAL_MANAGED_PROXY \
  --role=ACTIVE \
  --region=us-central1 \
  --network=shivaj-minimax-g4-384-cluster-net \
  --range=10.129.0.0/23 \
  --project=northam-ce-mlai-tpu
```

### 2.5 Configure IAM Permissions for GCS Bucket
Grant the necessary read permissions to your cluster's service accounts to access the model weights in the bucket:

```bash
# For the Workload Identity (Kubernetes Service Account)
gcloud storage buckets add-iam-policy-binding gs://shivaji-sglang-kimi-us-central1 \
  --member=serviceAccount:northam-ce-mlai-tpu.svc.id.goog[default/sglang-sa] \
  --role=roles/storage.objectViewer

# For the Default Compute Engine Service Account
gcloud storage buckets add-iam-policy-binding gs://shivaji-sglang-kimi-us-central1 \
  --member=serviceAccount:9452062936-compute@developer.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

### 2.6 Enable Workload Identity on GKE
Enabling Workload Identity is required for the GCS Fuse CSI driver. Run the following command to enable it:

```bash
gcloud container clusters update shivaji-minimax-g4-384-cluster \
  --workload-pool=northam-ce-mlai-tpu.svc.id.goog \
  --zone=us-central1-f \
  --project=northam-ce-mlai-tpu
```

### 3. Configure GCS Bucket and GCS Fuse
Ensure your cluster can access the model weights via GCS Fuse:

1. **Enable GCS Fuse CSI Driver** on your GKE cluster:
   ```bash
   gcloud container clusters update shivaji-minimax-g4-384-cluster \
  --update-addons=GcsFuseCsiDriver=ENABLED \
  --zone=us-central1-f \
  --project=northam-ce-mlai-tpu
   ```

2. **Verify Bucket and Weights**:
   Ensure the bucket exists and contains the `kimi-k2.6` model folder:
   ```bash
   # Check if bucket exists
   gcloud storage buckets describe gs://shivaji-sglang-kimi-us-central1
   
   # Check for model files
   gcloud storage ls gs://shivaji-sglang-kimi-us-central1/kimi-k2.6/
   ```

3. **Update Manifest**: The `bucketName` is already configured as `shivaji-sglang-kimi-us-central1` in `kimi-k2.6/sglang-kimik2.6-1node.yaml`.

### 4. Deploy SGLang Pods
Deploy the SGLang pod and metrics monitor configured for KimiK2.6:

```bash
kubectl apply -f kimi-k2.6/sglang-kimik2.6-1node.yaml
kubectl apply -f kimi-k2.6/sglang-kimi-podmonitoring.yaml
```

### 5. Deploy Endpoint Picker (EPP)
Install the EPP scheduler using the migrated `inference-gateway/epp-values.yaml` and the `v1.5.0` version of the chart from the public registry:

```bash
helm install precise-prefix-cache-routing \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.5.0 \
  -f inference-gateway/epp-values.yaml \
  -n default
```
*(Note: Running a `kubectl patch` command to strip deprecated metrics flags is **no longer needed** as `epp-values.yaml` bypasses flag injection by declaring `modelServerType: vllm` while EPP still correctly auto-detects SGLang pods via labels at runtime, and incorporates scraping interval optimization to prevent timeouts).*

### 6. Apply Gateway, HTTPRoute and InferenceObjective
Apply the manifests to create the load balancer, routing rules, and priority objective:

```bash
kubectl apply -f inference-gateway/gateway.yaml
kubectl apply -f inference-gateway/httproute.yaml
kubectl apply -f inference-gateway/inference-objective.yaml
```

Wait for the Gateway to be programmed and receive an IP address:
```bash
kubectl get gateway kimi-inference-gateway
```

### 7. Deploy Client App
Update the `VLLM_API_URL` environment variable in `inference-gateway/vllm-chat-app.yaml` to point to your Gateway's IP address, then apply it:

```bash
kubectl apply -f inference-gateway/vllm-chat-app.yaml
```

---

## Running Tests

All tests are designed to be run from *inside* the client pod (`vllm-chat-app-pod`) which has the required `openai` library installed.

### Copy Tests to Pod
Copy the test scripts from your local machine to the running pod:

```bash
kubectl cp inference-gateway/tests/ vllm-chat-app-pod:/app/
```

### Run Tests

1.  **Basic Functional Test:**
    ```bash
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/prefix_cache_benchmark.py
    ```

2.  **Concurrency Test (20 requests):**
    ```bash
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/prefix_cache_concurrency_test.py
    ```

3.  **Multi-Prefix Targeted Routing Test:**
    ```bash
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/prefix_specific_test.py
    ```

4.  **Unbalanced Load Test (10 vs 20):**
    ```bash
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/prefix_unbalanced_test.py
    ```

5.  **Stress Tests:**
    ```bash
    # 50 concurrent requests
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/high_concurrency_test.py
    # 100 concurrent requests
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/extreme_concurrency_test.py
    # 200 concurrent requests
    kubectl exec -it vllm-chat-app-pod -- python3 /app/tests/massive_stress_test.py
    ```

## Monitoring

To verify that routing is happening properly (e.g., all requests with Prefix A go to the same pod), monitor the logs of your SGLang pods in separate terminals, filtering out the noise:

```bash
kubectl logs <SGLANG_POD_NAME_0> -c sglang-container -f | grep -v -E "/metrics|/health"
```
*(Note: The current manifest deploys a StatefulSet `sglang-kimi-g4` with 2 replicas targeting `reflection-g4-pool`).*

---

## References

*   [GKE Inference Gateway Documentation](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api#inference-gateway)
*   [Gateway API Inference Extension GitHub](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

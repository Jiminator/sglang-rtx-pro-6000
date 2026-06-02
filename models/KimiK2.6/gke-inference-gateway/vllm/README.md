# GKE Inference Gateway: Prefix-Cache Aware Routing

This repository contains the configuration files and test scripts to set up and verify **Precise Prefix-Cache Aware Routing** using GKE Inference Gateway and **vLLM** for **KimiK2.6** model.  This setup ensures that requests sharing the same prompt prefix are routed to the same vLLM replica, maximizing KV cache reuse and reducing Time-to-First-Token (TTFT). 

## Folder Structure

*   `epp-values.yaml`: Helm values file for deploying the Endpoint Picker (EPP).
*   `gateway.yaml`: Kubernetes Gateway resource (using `gke-l7-rilb` for internal load balancing).
*   `httproute.yaml`: Kubernetes HTTPRoute resource to bind traffic to the InferencePool.
*   `inference-objective.yaml`: InferenceObjective resource for routing priority.
*   `tokenizer-patch-configmap.yaml`: Hotfix ConfigMap for custom tokenizers for KimiK2.6.
*   `vllm/`: vLLM deployment manifests for KimiK2.6 model servers.
*   `vllm-chat-app.yaml`: Deployment file for a lightweight Python CLI chat client.
*   `tokenizer-patch-configmap.yaml`: Hotfix ConfigMap for custom tokenizers (like Kimi).
*   `kimik2.6/`: vLLM deployment manifests for KimiK2.6 model servers.
    *   `vllm-kimi-g4.yaml`: StatefulSet manifest for vLLM replicas (configured for 2 replicas).
    *   `vllm-kimi-podmonitoring.yaml`: PodMonitoring resource for metrics.
*   `tests/`:
    *   `prefix_cache_benchmark.py`: Functional test for prefix caching (sequential).
    *   `prefix_cache_concurrency_test.py`: Test with 20 concurrent requests of the same prefix.
    *   `prefix_specific_test.py`: Test with two different prefixes to verify targeted routing.
    *   `prefix_unbalanced_test.py`: Test with unbalanced load (10 vs 20 requests) for two prefixes.
*   `vllm-chat-app.yaml`: Deployment file for a lightweight Python CLI chat client.

---

## Prerequisites

1.  **GKE Cluster**: Version 1.34+ recommended (for native InferencePool support).
2.  **HuggingFace Secret**: A Kubernetes secret named `hf-secret` containing your HuggingFace token:
    ```bash
    kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<your_token>
    ```
3.  **Model Weights**: A GCS bucket containing the KimiK2.6 model weights, referenced in `kimik2.6/vllm-kimi-g4.yaml` (or update the manifest to use your storage source).
4.  **InferenceExtension CRDs**: If your cluster version does not have them installed by default, install them manually:
    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.4.0/config/crd/bases/inference.networking.k8s.io_inferencepools.yaml
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.4.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml
    ```

---

## Step-by-Step Setup

### 1. Enable Gateway API on GKE
Ensure your cluster has the Gateway API enabled and HTTP Load Balancing active:

```bash
gcloud container clusters update <YOUR_CLUSTER_NAME> \
  --gateway-api=standard \
  --zone=<YOUR_ZONE> \
  --project=<YOUR_PROJECT_ID>
```

### 2. Create Proxy-Only Subnet (For Internal Load Balancing)
If you are using internal load balancing (as configured in `gateway.yaml` with `gke-l7-rilb`), you must create a proxy-only subnet in your VPC and region:

```bash
gcloud compute networks subnets create <SUBNET_NAME> \
  --purpose=REGIONAL_MANAGED_PROXY \
  --role=ACTIVE \
  --region=<YOUR_REGION> \
  --network=<YOUR_VPC_NAME> \
  --range=<IP_RANGE_e.g._172.16.1.0/24> \
  --project=<YOUR_PROJECT_ID>
```

### 3. Configure GCS Bucket and GCS Fuse
Ensure your cluster can access the model weights via GCS Fuse:

1. **Enable GCS Fuse CSI Driver** on your GKE cluster:
   ```bash
   gcloud container clusters update <YOUR_CLUSTER_NAME> \
     --update-addons=GcsFuseCsiDriver=ENABLED \
     --zone=<YOUR_ZONE> \
     --project=<YOUR_PROJECT_ID>
   ```

2. **Verify Bucket and Weights**:
   Ensure the bucket exists and contains the `kimi-k2.6` model folder. Ensure the bucket is in the same zone as the nodepool:
   ```bash
   # Check if bucket exists
   gcloud storage buckets describe gs://<YOUR_BUCKET_NAME>
   
   # Check for model files
   gcloud storage ls gs://<YOUR_BUCKET_NAME>/kimi-k2.6/
   ```

3. **Update Manifest**: Open `kimik2.6/vllm-kimi-g4.yaml` and update the `bucketName` field under the `models` volume to match your bucket.

### 4. Deploy vLLM Pods
Deploy the vLLM replicas configured for KimiK2.6 with ZeroMQ event streaming enabled:

```bash
kubectl apply -f kimik2.6/vllm-kimi-g4.yaml
kubectl apply -f kimik2.6/vllm-kimi-podmonitoring.yaml
```

### 5. Apply Tokenizer Patch (Required for Kimi models)
If using Kimi models, apply the tokenizer patch ConfigMap before deploying EPP:

```bash
kubectl apply -f tokenizer-patch-configmap.yaml
```

### 6. Deploy Endpoint Picker (EPP)
Install the EPP scheduler using the provided `epp-values.yaml`. This assumes you have access to the GKE Inference Gateway Helm chart registry:

```bash
helm install precise-prefix-cache-routing \
  oci://us-central1-docker.pkg.dev/cloud-ai-gke/gke-inference-gateway/charts/inferencepool \
  -f epp-values.yaml \
  -n default
```

### 7. Apply Gateway, HTTPRoute and InferenceObjective
Apply the manifests to create the load balancer, routing rules, and priority objective:

```bash
kubectl apply -f gateway.yaml
kubectl apply -f httproute.yaml
kubectl apply -f inference-objective.yaml
```

Wait for the Gateway to be programmed and receive an IP address:
```bash
kubectl get gateway kimi-inference-gateway
```

### 8. Deploy Client App
Update the `VLLM_API_URL` environment variable in `vllm-chat-app.yaml` to point to your Gateway's IP address, then apply it:

```bash
kubectl apply -f vllm-chat-app.yaml
```

---

## Running Tests

All tests are designed to be run from *inside* the client pod (`vllm-chat-app-pod`) which has the required `openai` library installed.

### Copy Tests to Pod
Copy the test scripts from your local machine to the running pod:

```bash
kubectl cp tests/ vllm-chat-app-pod:/app/
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

## Monitoring

To verify that routing is happening properly (e.g., all requests with Prefix A go to the same pod), monitor the logs of your vLLM pods in separate terminals, filtering out the noise:

```bash
kubectl logs <VLLM_POD_NAME_0> -c vllm-container -f | grep -v -E "/metrics|/health"
kubectl logs <VLLM_POD_NAME_1> -c vllm-container -f | grep -v -E "/metrics|/health"
```

---

## References

*   [GKE Inference Gateway Documentation](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api#inference-gateway)
*   [Gateway API Inference Extension GitHub](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

# **\[External\] Global Multi-Region SGLang \+ Kimi k2.5 \+ GKE Inference Gateway on G4 Setup Guide**

Author: Sina Chavoshi Date: Apr 17, 2026

## **Introduction**

Deploying massive large language models like Kimi k2.5 globally requires a robust architecture capable of intelligent traffic routing, massive model offloading, and reliable failover. The GKE Multi-Cluster Inference Gateway acts as a unified entry point, seamlessly load-balancing inference requests across multiple regional clusters. When paired with SGLang for distributed inference, this architecture ensures that Kimi k2.5 requests are efficiently distributed based on intelligent metrics. Based on our extensive testing see [benchmark results](#bookmark=id.lhqv2x547572), we have identified several critical configurations required to run the Kimi k2.5 model reliably at high concurrency across a multi-cluster G4 fleet.

## **Key Takeaways & Lessons Learned**

* **Proxy-Only Subnet Constraints:** Google Cloud enforces a strict limit of only one proxy-only subnet per region per VPC network. You cannot have both regional and global managed proxy subnets in the same region.  
* **VPC Restrictions:** All clusters in the fleet must be deployed in the same VPC. Cross-VPC load balancing is not supported for multi-cluster Gateways.  
* **SGLang Master Pod Targeting:** In SGLang's distributed mode, only the master pod (rank 0\) serves the API. Routing traffic to worker pods results in 404 Not Found errors.  
* **Gateway Timeouts:** The default Google Cloud Load Balancer timeout aggressively drops long-running inference requests under heavy load. The backend policy must be extended to 1 hour (3600 seconds).

## **Reference Documentation**

For more granular configuration options, refer to the official Google Cloud documentation and internal tracking documents:

* **Setting up L7 Cross-Region Internal Load Balancing:** docs.cloud.google.com/load-balancing/docs/l7-internal/setting-up-l7-cross-reg-internal  
* **Customize Backend Multi-Cluster Inference Gateway:** docs.cloud.google.com/kubernetes-engine/docs/how-to/customize-backend-multicluster-inference-gateway  
* **Deploy Resources to Target Clusters:** docs.cloud.google.com/kubernetes-engine/docs/how-to/setup-multicluster-inference-gateway\#deploy-resources-to-target-clusters  
* **About Multi-Cluster Inference Gateway Concepts:** docs.cloud.google.com/kubernetes-engine/docs/concepts/about-multi-cluster-inference-gateway

### **1\. Network Configuration: Global Managed Proxy Subnets**

The cross-regional load balancer requires proxy-only subnets with purpose=GLOBAL\_MANAGED\_PROXY. Critical Constraint: If a regional proxy subnet already exists in your target region, you must delete it first. Run the following command for each target region (e.g., us-east5 and us-west8), replacing YOUR\_VPC\_NETWORK with your actual VPC name:

```shell
gcloud compute networks subnets create global-proxy-subnet-us-east5 \
 --purpose=GLOBAL_MANAGED_PROXY \
 --role=ACTIVE \
 --region=us-east5 \
 --network=YOUR_VPC_NETWORK \
 --range=10.124.0.0/23
gcloud compute networks subnets create global-proxy-subnet-us-west8 \
 --purpose=GLOBAL_MANAGED_PROXY \
 --role=ACTIVE \
 --region=us-west8 \
 --network=YOUR_VPC_NETWORK \
 --range=10.126.0.0/23
```

### **2\. Cluster & G4 Node Pool Creation**

Create your GKE clusters with Advanced Datapath enabled, followed by the G4 GPU pools. Repeat these commands for each region you intend to deploy in. Create the Cluster (Example: us-east5): Note: This command enables Advanced Datapath (required for multi-NIC G4 performance), Workload Identity, DCGM GPU monitoring, and automatically registers the cluster to your fleet.

```shell
gcloud beta container \
 --project "YOUR_PROJECT_ID" \
 clusters create "kimi-cluster-us-east5-a" \
 --zone "us-east5-a" \
 --no-enable-basic-auth \
 --cluster-version "1.35.1-gke.1396002" \
 --release-channel "regular" \
 --machine-type "g4-standard-384" \
 --accelerator "type=nvidia-rtx-pro-6000,count=8" \
 --image-type "COS_CONTAINERD" \
 --disk-type "hyperdisk-balanced" \
 --disk-size "100" \
 --metadata disable-legacy-endpoints=true \
 --service-account "default" \
 --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/trace.append" \
 --max-pods-per-node "110" \
 --spot \
 --num-nodes "2" \
 --logging=SYSTEM,WORKLOAD \
 --monitoring=SYSTEM,DEPLOYMENT,STATEFULSET,JOBSET,STORAGE,HPA,POD,DAEMONSET,CADVISOR,KUBELET,DCGM \
 --enable-ip-alias \
 --network "projects/YOUR_PROJECT_ID/global/networks/YOUR_VPC_NETWORK" \
 --subnetwork "projects/YOUR_PROJECT_ID/regions/us-east5/subnetworks/default" \
 --no-enable-intra-node-visibility \
 --default-max-pods-per-node "110" \
 --enable-ip-access \
 --security-posture=standard \
 --workload-vulnerability-scanning=disabled \
 --no-enable-google-cloud-access \
 --addons HorizontalPodAutoscaling,HttpLoadBalancing,NodeLocalDNS,GcePersistentDiskCsiDriver \
 --enable-autoupgrade \
 --enable-autorepair \
 --max-surge-upgrade 1 \
 --max-unavailable-upgrade 0 \
 --enable-managed-prometheus \
 --workload-pool "YOUR_PROJECT_ID.svc.id.goog" \
 --enable-shielded-nodes \
 --shielded-integrity-monitoring \
 --no-shielded-secure-boot \
 --fleet-project="YOUR_PROJECT_ID" \
 --node-locations "us-east5-a" \
 --enable-multi-networking \
 --gateway-api=standard \
 --ephemeral-storage-local-ssd count=32 \
 --datapath-provider=ADVANCED_DATAPATH
```

### **3\. Fleet Registration & Gateway Enablement**

To route traffic globally, clusters must be registered to a fleet. Register both clusters to the fleet:

```shell
gcloud container fleet memberships register kimi-cluster-us-east5-a \
 --gke-cluster us-east5-a/kimi-cluster-us-east5-a \
 --project YOUR_PROJECT_ID
gcloud container fleet memberships register kimi-cluster-us-west8-a \
 --gke-cluster us-west8-a/kimi-cluster-us-west8-a \
 --project YOUR_PROJECT_ID
```

Enable Multi-Cluster Ingress on the Config Cluster: Designate one cluster (e.g., us-east5) as the configuration hub where the Gateway resources will live.

```shell
gcloud container fleet ingress enable \
 --config-membership kimi-cluster-us-east5-a \
 --location us-east5-a \
 --project YOUR_PROJECT_ID
```

### **4\. InferencePool Deployment (Helm Method)**

Deploy the InferencePool to all clusters using Helm. This method automatically configures the pool and EPP. Run the following Helm command on all clusters (adjusting context and names as needed):

```shell
helm install kimi-sglang-pool \
 --kube-context CLUSTER_CONTEXT \
 --set inferencePool.modelServers.matchLabels.app=distributed-sglang \
 --set "inferencePool.modelServers.matchLabels.apps\.kubernetes\.io/pod-index=0" \
 --set inferencePool.targetPortNumber=30000 \
 --set provider.name=gke \
 --set inferenceExtension.monitoring.gke.enabled=true \
 --version v1.1.0 \
 oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
```

**\[\!IMPORTANT\] Port Configuration Fix:** In version v1.1.0 of the chart, the \--set inferencePool.targetPortNumber=30000 flag may be ignored, defaulting the port to 8000\. If this happens, manually patch the InferencePool to use port 30000 (required for SGLang) before or after exporting:

```shell
kubectl patch inferencepool kimi-sglang-pool --type=merge -p '{"spec":{"targetPorts":[{"number":30000}]}}' --context=CLUSTER_CONTEXT
```

And annotate the InferencePool to be exported:

```shell
kubectl annotate inferencepool kimi-sglang-pool networking.gke.io/export="True" --context=CLUSTER_CONTEXT
```

### **6\. Gateway & Global Routing Configuration**

Apply the following manifests ONLY to your Config Cluster (kimi-cluster-us-east5-a). This provisions the load balancer and enforces the extended 1-hour backend connection timeout necessary for long-context inferences. Save this as gateway-config.yaml and run kubectl apply \-f gateway-config.yaml:

```
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: cross-region-kimi-gateway
  namespace: default
spec:
  gatewayClassName: gke-l7-cross-regional-internal-managed-mc
  addresses:
  - type: networking.gke.io/ephemeral-ipv4-address/us-east5
    value: "us-east5"
  - type: networking.gke.io/ephemeral-ipv4-address/us-west8
    value: "us-west8"
  listeners:
  - name: http
    protocol: HTTP
    port: 80
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: sglang-kimi-pool-default
  namespace: default
spec:
  parentRefs:
  - name: cross-region-kimi-gateway
    kind: Gateway
  rules:
  - backendRefs:
    - group: networking.gke.io
      kind: GCPInferencePoolImport
      name: sglang-kimi-pool
      port: 30000
---
apiVersion: networking.gke.io/v1
kind: GCPBackendPolicy
metadata:
  name: kimi-sglang-pool
  namespace: default
spec:
  default:
    timeoutSec: 3600
    balancingMode: CUSTOM_METRICS
    trafficDuration: LONG
    customMetrics:
    - name: gke.named_metrics.kv-cache
      dryRun: false
      maxUtilizationPercent: 60
  targetRef:
    group: "networking.gke.io"
    kind: GCPInferencePoolImport
    name: sglang-kimi-pool
```

### **7\. Verification & Header Requirements**

If you are you using inference objective , to successfully route traffic through the gateway and leverage the Endpoint Picker Proxy (EPP) custom metrics:

* Client requests must pass the header: x-gateway-inference-objective: kimi-objective.

## **Performance Benchmarks (Kimi k2.5 Model)**

Extensive load testing was conducted against single-cluster and multi-cluster configurations using n2-standard-32 CPU nodes to drive the traffic. The Inference Gateway introduces zero significant overhead compared to direct API calls, and successfully triggers regional spillover under high concurrency.

### **Benchmark Results Table**

All tests run with \--seed 42 and \--random-input 1024\. Few relevant results are as follows, run\#6 & \#7 compare direct benchmark against cluster v.s. Via Inference gateway with a single instance adds no additional impact.  Run\#11 is the highest performance with a single server behind the inference gateway. Run\# 24 shows the same performance linearly scales with two clusters across two regions. 

| Run \# | Cluster | Target | Prompts | Concurrency | Success Rate | Request Throughput (req/s) | Token Throughput (tok/s) | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 6 | us-east5-a | Direct | 500 | 384 | 500 / 500 | 0.61 | 2480 | Testing midpoint concurrency. Succeeded with high throughput\! |
| 7 | us-east5-a | Gateway | 500 | 384 | 500 / 500 | 0.60 | 2467 | Midpoint concurrency via Gateway. Very close to Direct performance\! |
| 11 | us-east5-a | Gateway | 1536 | 448 | 1534 / 1536 | 0.72 | 2898 | Full scale run\! 99.87% success rate. High throughput close to baseline\! |
| 24 | us-east5-a us-west8-a | Gateway | 2x2000 | 2x440 | 3998 / 4000 | 1.40 | 6380 | Dual-Pod Multi-Client scaling test completed with aggregate throughput\! |
| 25 | us-east5-a, us-west8-a, europe-west4-b | Gateway | 3x2000 | 3x420 | 5994 / 6000 | 2.10 | 8457 | 3-Pod Multi-Client setup across 3 regions\! Achieved aggregate 2.1 req/s throughput. All requests generated from us-east-5 hitting the same endpoint and routed to 3 regions.   |

### **8\. Required SGLang Custom Metrics Configuration (CRITICAL FIXES)**

To enable dynamic KV-cache routing specifically for SGLang and fully leverage the Endpoint Picker Proxy (EPP), ensure the following three configurations are deployed across the active multi-cluster mesh:

#### 8.1 Upgrade Endpoint Picker Proxy to v1.5.0-rc.2

The base v1.1.0 EPP image does not support advanced custom metric extraction for SGLang. Upgrade the proxy deployments directly on all active clusters using:

```shell
kubectl set image deployment/sglang-kimi-pool-epp epp=registry.k8s.io/gateway-api-inference-extension/epp:v1.5.0-rc.2
```

#### 8.2 Correct SGLang KV Cache Telemetry Mapping

Unlike vLLM which exposes an absolute KV-cache percentage metric, SGLang exposes KV-cache token utilization under the metric sglang:token\_usage. Update your AutoscalingMetric query mapping to reflect this:

```
apiVersion: autoscaling.gke.io/v1beta1
kind: AutoscalingMetric
metadata:
  name: sglang-kv-cache
  namespace: default
spec:
  metrics:
  - pod:
      containers:
      - endpoint:
          path: /metrics
          port: 9090
        metrics:
        - gauge:
            loadBalancing:
              enabled: true
            name: kv-cache
            prometheusMetricName: inference_pool_average_kv_cache_utilization
      selector:
        matchLabels:
          inferencepool: sglang-kimi-pool-epp
```

#### 8.3 Deploy PodMonitoring to Scrape EPP Proxy

Deploy a PodMonitoring resource targeting the EPP proxy on port 9090 so custom scoring telemetry is automatically ingested into the Google Cloud Monitoring dashboards:

```
apiVersion: monitoring.googleapis.com/v1
kind: PodMonitoring
metadata:
  name: sglang-kimi-pool-epp
  namespace: default
  labels:
    app: sglang-kimi-pool-epp
spec:
  selector:
    matchLabels:
      inferencepool: sglang-kimi-pool-epp
  endpoints:
  - port: 9090
    path: /metrics
    interval: 15s
    scheme: http
```

# **Meeting follow ups:**

1. Observability   
2. Fine tuning the routing  \- \> see [https://docs.cloud.google.com/kubernetes-engine/docs/how-to/customize-backend-multicluster-inference-gateway\#supported-fields](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/customize-backend-multicluster-inference-gateway#supported-fields)  
3. How to add Dynamo metrics and convert them 


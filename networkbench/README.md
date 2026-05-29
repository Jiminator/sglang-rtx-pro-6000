# GKE G4 High-Bandwidth Networking & NCCL Benchmarks

This directory contains scripts and Kubernetes manifests to set up a GKE cluster with high-bandwidth networking (DRANET) on G4 instances (NVIDIA RTX PRO 6000) and run network benchmarks using `iperf3` and NCCL.

## Folder Contents

-   **Documentation**:
    -   `Network.md`: Original guide for GKE G4 setup.
    -   `Iperf+dra`: Snippet for iperf deployment and test results.
-   **Setup Scripts**:
    -   `01_setup_network.sh`: Creates the dedicated 8896 MTU VPC and subnets.
    -   `02_create_cluster.sh`: Creates the GKE cluster and G4 node pool.
-   **Kubernetes Manifests**:
    -   `03_deploy_dra.yaml`: `ResourceClaimTemplate` for Dynamic Resource Allocation (DRA) of network devices.
    -   `04_iperf_dranet.yaml`: Deployment for `iperf3` bandwidth testing.
    -   `05_nccl_allreduce.yaml`: Deployment for initial NCCL AllReduce test (single NIC).
    -   `chiv01-nccl-bench.yaml`: Optimized deployment for NCCL AllReduce using both NICs (Multi-NIC).
-   **Benchmark Scripts**:
    -   `run_bench_10x.sh`: Runs the NCCL benchmark 10 times and logs results.
    -   `run_iperf_10x.sh`: Runs the `iperf3` bidirectional test 10 times and logs results.
-   **Logs**:
    -   `run_*.log`: Logs from NCCL benchmark runs.
    -   `run_iperf_*.log`: Logs from `iperf3` runs.

## Steps to Run NCCL Benchmarks

### 1. Infrastructure Setup
Run the scripts to create the network and cluster. These scripts include checks to skip creation if the resource already exists.

```bash
./01_setup_network.sh
./02_create_cluster.sh
```

### 2. Deploy DRA Template
Apply the ResourceClaimTemplate required for high-bandwidth networking.

```bash
kubectl apply -f 03_deploy_dra.yaml
```

### 3. Run NCCL Benchmark
To run the optimized multi-NIC NCCL benchmark:

1.  **Apply the manifest**:
    ```bash
    kubectl apply -f chiv01-nccl-bench.yaml
    ```
2.  **Wait for pods to be Running**:
    ```bash
    kubectl get pods -l app=nccl-golden-custom
    ```
3.  **Run the benchmark**:
    You can use the automated script (ensure pod names in the script match the running pods):
    ```bash
    ./run_bench_10x.sh
    ```
    Or run manually on both pods (substitute actual pod names and Master IP):
    *   **Rank 0**:
        ```bash
        kubectl exec <POD_RANK_0> -- bash -c "MASTER_ADDR=<MASTER_IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python3 /root/dist_test.py"
        ```
    *   **Rank 1**:
        ```bash
        kubectl exec <POD_RANK_1> -- bash -c "MASTER_ADDR=<MASTER_IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python3 /root/dist_test.py"
        ```

## Summary of Test Results

### 1. Iperf3 Bidirectional Test (8 Streams)
-   **Transmit (TX)**: **190 Gbits/sec** (consistent across 10 runs)
-   **Receive (RX)**: **190 Gbits/sec** (consistent across 10 runs)
-   *Note: This test saturated a single 200 Gbps NIC in full-duplex mode.*

### 2. NCCL AllReduce (Multi-NIC Golden Setup)
-   **Payload Size**: 512.0 MB
-   **Median Measured Aggregate AlgBW**: **37.925 GB/s** (~303 Gbits/sec) over 10 runs.
-   *Note: This setup successfully utilized both `eth1` and `eth2` interfaces for higher aggregate bandwidth.*

### 3. NCCL AllReduce (Multiple Sizes, Multi-NIC)
Testing scaling with different payload sizes:

| Size (MB) | Time (us) | AlgBW (GB/s) | Approx. Bitrate (Gbps) |
| :--- | :--- | :--- | :--- |
| 256.0 | 15078.72 | 35.60 | ~284.8 |
| 512.0 | 29465.29 | 36.44 | ~291.5 |
| 1024.0 | 56750.69 | 37.84 | ~302.7 |
| 2048.0 | 112527.62 | 38.17 | ~305.4 |

*Note: Bandwidth increases with payload size, approaching ~38.17 GB/s for the largest size tested.*

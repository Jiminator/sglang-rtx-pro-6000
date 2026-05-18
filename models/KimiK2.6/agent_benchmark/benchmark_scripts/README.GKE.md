# How to Build and Run Agentic Benchmark

This guide outlines the steps to build the Docker image for the agentic benchmark and run it as a Kubernetes Job on GKE.

## Project Structure
Located in `/Users/shivajid/code/sglang/kimi/AgentBenchmark/`:
- `Dockerfile`: Builds the benchmark image by extending a base image and copying the specific test script.
- `agentic_benchmark_sglang_low_load.py`: The benchmark script that simulates low load.
- `agentic-benchmark-job-cpu-pool.yaml`: The Kubernetes Job manifest.

## Step 1: Build and Push the Docker Image

You need to build the Docker image and push it to your Google Artifact Registry.

1. Navigate to the `AgentBenchmark` directory:
   ```bash
   cd /Users/shivajid/code/sglang/kimi/AgentBenchmark
   ```

2. Build the Docker image:
   ```bash
   docker build -t us-central1-docker.pkg.dev/northam-ce-mlai-tpu/shivaji-rflx-docker-repo/agentic-benchmark:low-load .
   ```

3. Push the image to Artifact Registry:
   ```bash
   docker push us-central1-docker.pkg.dev/northam-ce-mlai-tpu/shivaji-rflx-docker-repo/agentic-benchmark:low-load
   ```

## Step 2: Run the Benchmark Job

1. Verify or update the target SGLang server URL in `agentic-benchmark-job-cpu-pool.yaml`. Line 13 contains the arguments:
   ```yaml
   args: ["http://10.10.0.7:30000", "--parallelism", "64", "--timeout", "1800.0"]
   ```
   Update `10.10.0.7:30000` to your actual SGLang server IP and port if it has changed.

2. Deploy the Job to the cluster:
   ```bash
   kubectl apply -f /Users/shivajid/code/sglang/kimi/AgentBenchmark/agentic-benchmark-job-cpu-pool.yaml
   ```

## Step 3: Monitor the Benchmark

1. Check the status of the job:
   ```bash
   kubectl get jobs
   ```

2. View the logs of the benchmark pod to see progress:
   ```bash
   kubectl logs -f -l job-name=agentic-benchmark-job-hicahche-cpu-pool-low-load
   ```

The logs will show `Success` messages for completed steps. At the end of the run, it will print a large JSON block with `=== BENCHMARK RESULTS ===`.


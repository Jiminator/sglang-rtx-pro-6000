#!/bin/bash
# Script to provision the GKE Cluster and DRANET Node Pool

set -e

PROJECT_ID=${PROJECT_ID:-"northam-ce-mlai-tpu"}
ZONE="us-central1-b"
CLUSTER_NAME="chavoshi-v10-cluster"
NETWORK_NAME="dranet-v10-vpc"

echo "Using project: $PROJECT_ID"

if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Cluster $CLUSTER_NAME already exists. Skipping creation."
else
    echo "[Step 1/2] Creating the Pure Dataplane-V2 Cluster: $CLUSTER_NAME..."
    gcloud container clusters create "$CLUSTER_NAME" \
        --network="$NETWORK_NAME" \
        --subnetwork=v10-sub0 \
        --enable-ip-alias \
        --enable-dataplane-v2 \
        --zone="$ZONE" \
        --num-nodes=1 \
        --machine-type=e2-standard-4 \
        --project="$PROJECT_ID"
    echo "Cluster $CLUSTER_NAME created."
fi

if gcloud container node-pools describe chavoshi-v10-pool --cluster="$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Node pool chavoshi-v10-pool already exists. Skipping creation."
else
    echo "[Step 2/2] Creating the G4 compute pool: chavoshi-v10-pool..."
    gcloud container node-pools create chavoshi-v10-pool \
        --cluster="$CLUSTER_NAME" \
        --machine-type=g4-standard-384 \
        --accelerator type=nvidia-rtx-pro-6000,count=8 \
        --enable-gvnic \
        --spot \
        --num-nodes=2 \
        --zone="$ZONE" \
        --node-labels=cloud.google.com/gke-networking-dra-driver=true \
        --additional-node-network network="$NETWORK_NAME",subnetwork=v10-sub1 \
        --additional-node-network network="$NETWORK_NAME",subnetwork=v10-sub2 \
        --project="$PROJECT_ID"
    echo "Node pool chavoshi-v10-pool created."
fi

echo "Cluster and node pool creation complete."

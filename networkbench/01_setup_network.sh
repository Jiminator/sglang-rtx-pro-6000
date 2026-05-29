#!/bin/bash
# Script to create the High-Bandwidth DRANET VPC and Subnets

set -e

PROJECT_ID=${PROJECT_ID:-"northam-ce-mlai-tpu"}
REGION="us-central1"
NETWORK_NAME="dranet-v10-vpc"

echo "Using project: $PROJECT_ID"

if gcloud compute networks describe "$NETWORK_NAME" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "VPC $NETWORK_NAME already exists. Skipping creation."
else
    echo "[Step 1/5] Creating the Dedicated 8896 MTU VPC: $NETWORK_NAME..."
    gcloud compute networks create "$NETWORK_NAME" \
        --subnet-mode=custom \
        --mtu=8896 \
        --project="$PROJECT_ID"
    echo "VPC $NETWORK_NAME created."
fi

if gcloud compute networks subnets describe v10-sub0 --region="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Subnet v10-sub0 already exists. Skipping creation."
else
    echo "[Step 2/5] Creating subnet v10-sub0 (Control)..."
    gcloud compute networks subnets create v10-sub0 \
        --network="$NETWORK_NAME" \
        --range=10.10.0.0/16 \
        --region="$REGION" \
        --project="$PROJECT_ID"
    echo "Subnet v10-sub0 created."
fi

if gcloud compute networks subnets describe v10-sub1 --region="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Subnet v10-sub1 already exists. Skipping creation."
else
    echo "[Step 3/5] Creating subnet v10-sub1 (Payload 1)..."
    gcloud compute networks subnets create v10-sub1 \
        --network="$NETWORK_NAME" \
        --range=10.20.0.0/16 \
        --region="$REGION" \
        --project="$PROJECT_ID"
    echo "Subnet v10-sub1 created."
fi

if gcloud compute networks subnets describe v10-sub2 --region="$REGION" --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Subnet v10-sub2 already exists. Skipping creation."
else
    echo "[Step 4/5] Creating subnet v10-sub2 (Payload 2)..."
    gcloud compute networks subnets create v10-sub2 \
        --network="$NETWORK_NAME" \
        --range=10.30.0.0/16 \
        --region="$REGION" \
        --project="$PROJECT_ID"
    echo "Subnet v10-sub2 created."
fi

if gcloud compute firewall-rules describe allow-dranet-internal --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "Firewall rule allow-dranet-internal already exists. Skipping creation."
else
    echo "[Step 5/5] Opening Internal VPC Firewall..."
    gcloud compute firewall-rules create allow-dranet-internal \
        --network="$NETWORK_NAME" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.0.0.0/8 \
        --project="$PROJECT_ID"
    echo "Firewall rule created."
fi

echo "Network setup complete."

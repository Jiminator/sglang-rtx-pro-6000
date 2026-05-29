# **GKE G4 High-Bandwidth Setup & Benchmarking Guide**

**Version:** 9.2  
**Authors:** [Sina Chavoshi](mailto:chavoshi@google.com)[Gaurav Ghildiyal](mailto:gauravkg@google.com)   
**Hardware Target:** G4 Instances (NVIDIA RTX PRO 6000, 2x 200 Gbps Diorite SmartNICs)

## **Overview**

This guide outlines the recommended production architecture for extracting maximum high-bandwidth multi-NIC network performance on G4 instances in Google Kubernetes Engine (GKE). It utilizes the **8896-MTU DRANET API Plane**, delivering the raw 23.5+ GB/s hardware ceiling natively using modern container device scheduling.

---

## **Part 1: Cluster & Infrastructure Setup**

### **1\. Create the High-Bandwidth DRANET VPC & Subnets**

```shell
# Create the Dedicated 8896 MTU VPC
gcloud compute networks create dranet-v10-vpc --subnet-mode=custom --mtu=8896 --project=[PROJECT_NAME]

# Create Isolated Subnets for Control (0) and Payloads (1 & 2)
gcloud compute networks subnets create v10-sub0 --network=dranet-v10-vpc --range=10.10.0.0/16 --region=us-central1 --project=[PROJECT_NAME]
gcloud compute networks subnets create v10-sub1 --network=dranet-v10-vpc --range=10.20.0.0/16 --region=us-central1 --project=[PROJECT_NAME]
gcloud compute networks subnets create v10-sub2 --network=dranet-v10-vpc --range=10.30.0.0/16 --region=us-central1 --project=[PROJECT_NAME]

# Open Internal VPC Firewall
gcloud compute firewall-rules create allow-dranet-internal \
    --network=dranet-v10-vpc --allow=tcp,udp,icmp --source-ranges=10.0.0.0/8 --project=[PROJECT_NAME]
```

### **2\. Provision the GKE Cluster & DRANET Node Pool**

```shell
# Create the Pure Dataplane-V2 Cluster (Note: --enable-multi-networking is NOT required for the DRANET plane)Create a cluster with Dataplane V2 enabled
gcloud container clusters create chavoshi-v10-cluster \
    --network=dranet-v10-vpc --subnetwork=v10-sub0 --enable-ip-alias --enable-dataplane-v2 \
    --zone=us-central1-b --num-nodes=1 --machine-type=e2-standard-4 --project=[PROJECT_NAME]

# Create the G4 compute pool with the GKE DRANET enabledDRA Device Driver active
gcloud container node-pools create chavoshi-v10-pool \
    --cluster=chavoshi-v10-cluster --machine-type=g4-standard-384 \
    --accelerator type=nvidia-rtx-pro-6000,count=8 --enable-gvnic --spot --num-nodes=2 --zone=us-central1-b \
    --node-labels=cloud.google.com/gke-networking-dra-driver=true \
    --additional-node-network network=dranet-v10-vpc,subnetwork=v10-sub1 \
    --additional-node-network network=dranet-v10-vpc,subnetwork=v10-sub2 --project=[PROJECT_NAME]
```

### **3\. Deploy Kubernetes Dynamic Resource Allocation Templates**

```
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: netdev-claim-template
spec:
  spec:
    devices:
      requests:
      - name: netdev-request
        exactly:
          deviceClassName: netdev.google.com
          allocationMode: ExactCount
          count: 2
```

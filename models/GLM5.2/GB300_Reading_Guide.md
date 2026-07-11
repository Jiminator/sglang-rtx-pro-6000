# GB300 / NVL72 Architecture & Software: A Reading Guide

A curated, layered map of the authoritative documentation for the GB300 NVL72
rack-scale architecture and its software stack (Fabric Manager, NVLSM, NMX,
IMEX, partitions), oriented toward standing up multi-node NVLink (MNNVL)
workloads such as prefill/decode disaggregated serving.

---

## The mental model (read this first)

NVLink stopped being an intra-node bus and became a **rack-scale network**. With
fourth-generation NVSwitch, up to **72 GPUs form a single NVLink domain**, and by
default the entire NVL72 is **one NVLink partition** in which every GPU can
directly access every other GPU's memory.

That capability introduces two new software layers:

1. **A fabric control plane on the switch trays** — Fabric Manager (FM) + NVLink
   Subnet Manager (NVLSM) + NMX services, running on NVOS — which routes and
   partitions the fabric.
2. **A per-node service, IMEX** (Internode Memory Exchange/Management) — which
   lets CUDA processes in *different OS domains* securely export and import each
   other's GPU memory over that fabric.

Everything you run into operationally (`nodes_config.cfg`, `nvidia-imex-ctl`,
`/dev/nvidia-caps-imex-channels/channel0`, the `Fabric` block in
`nvidia-smi -q`) is a surface of the **IMEX + fabric** layer.

---

## Quick reference: the files & tools you actually touch

| Artifact | What it is | Where it's documented |
|---|---|---|
| `nvidia-smi -q` → `Fabric` block (`State`, `ClusterUUID`, `CliqueId`) | Whether a GPU has joined the NVLink fabric and which partition/clique it's in | IMEX overview + Partition guide |
| `nvidia-imex` (systemd service) | Per-node daemon that orchestrates cross-node GPU memory export/import | IMEX Getting Started |
| `/etc/nvidia-imex/config.cfg` | IMEX service config; sets `IMEX_NODE_CONFIG_FILE`, ports, logging | IMEX Config Options |
| `/etc/nvidia-imex/nodes_config.cfg` | List of node IPs in the IMEX domain (default filename; overridable) | IMEX Config Options |
| `nvidia-imex-ctl` | Query IMEX domain connectivity (`C` = connected) | IMEX Getting Started |
| `/dev/nvidia-caps-imex-channels/channelN` | Char device granting a process access to an IMEX channel | IMEX Channels |
| `NVreg_CreateImexChannel0` (module param) | Auto-create `channel0` at driver load | IMEX Channels |
| `nv show sdn partition` (switch CLI) | Inspect/manage NVLink partitions on the switch tray | Mission Control fabric mgmt |

---

## Start here — the hub

- **NVIDIA Multi-Node NVLink Systems (index)** — the landing page that links the
  IMEX guide, tuning guide, partition guide, and the debug/firmware tools.
  <https://docs.nvidia.com/multi-node-nvlink-systems/index.html>

---

## Layer 1 — The multi-node memory-sharing model + IMEX

The layer you interact with most directly. A CUDA process allocates GPU memory
and obtains a shareable handle, which triggers a virtual → physical → **fabric**
address mapping; IMEX orchestrates that mapping across nodes over TCP/gRPC.

- **Overview / memory-sharing model**
  <https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html>
- **Getting Started** (install, start the service, verify)
  <https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/gettingstarted.html>
- **Config Options** (documents `IMEX_NODE_CONFIG_FILE` and its default — i.e.
  why the node-list filename matters)
  <https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/config.html>
- **IMEX Channels** (`/dev/nvidia-caps-imex-channels`, `mknod`,
  `NVreg_CreateImexChannel0`, multi-user isolation)
  <https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html>
- **Deployment Models**
  <https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/deployment.html>

---

## Layer 2 — The fabric control plane (switch-tray software)

Fabric Manager configures the NVSwitch memory fabric and coordinates with NVLSM,
which programs switch routing tables and partition keys. This runs on the
NVLink switch trays under NVOS.

- **NVIDIA Fabric Manager User Guide** (FM responsibilities, NVLSM interaction,
  NVSwitch generations)
  <https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html>
- **DGX GB Rack Scale Systems — Software chapter** (best single summary of how
  FM, NVLSM, NMX-C/NMX-T, NVOS, IMEX, DCGM, and NCCL fit together)
  <https://docs.nvidia.com/dgx/dgxgb200-user-guide/software.html>
- **DGX GB Rack Scale Systems — Networking chapter** (physical/network layout,
  interface names, the networks used by fabric services)
  <https://docs.nvidia.com/dgx/dgxgb200-user-guide/networking.html>

---

## Layer 3 — NVLink partitioning (why "same NVLink domain" is a real constraint)

The NVL72 is one default partition, but admins can carve **User Partitions**.
Two nodes get MNNVL only if they share the same partition/clique — so partition
state is the first thing to confirm when cross-node NVLink "should" work but
doesn't.

- **GB200 NVL Partition User Guide** (partition types, admin vs tenant views of
  NVLink and GPU fabric state)
  <https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf>
- **Mission Control — High-Speed Fabric Management** (the `nv show sdn partition`
  switch CLI, partition create/delete/reroute)
  <https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.0.0/high-speed-fabric-management.html>

---

## Layer 4 — Tuning, validation & debug tooling

- **Multi-Node Tuning Guide** (performance, system/network model)
  <https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html>
- **DCGM Multi-Node Diagnostics** (fabric health checks; explicitly depend on
  IMEX being configured — a good end-to-end validation)
  <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-multinode-diagnostics.html>
- **NVDebug** and **NVFWUPD** (log collection and firmware update) — both linked
  from the Multi-Node NVLink hub above.

---

## Your platform — GCP A4X Max (GB300)

On GCP, a GB300 node is an **A4X Max** instance
(`a4x-maxgpu-4g-metal`, `nvidia-gb300`). The NVLink fabric spans a **subblock of
18 instances (72 GPUs)**; cross-subblock traffic falls back to RoCE. Two nodes
share MNNVL only if reserved in the same subblock.

- **GCP GPU network bandwidth / architecture** (A4X vs A4X Max, NVLink subblock
  vs RoCE inter-subblock)
  <https://docs.cloud.google.com/compute/docs/gpus/gpu-network-bandwidth>
- **GCP A4X Max on GKE** (on GKE you do **not** hand-edit IMEX — the NVIDIA DRA
  driver's ComputeDomains manage IMEX domains/channels for you)
  <https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max>
- **NVIDIA: Enabling Multi-Node NVLink on Kubernetes** (how ComputeDomains wrap
  IMEX; useful even on bare VMs for understanding what IMEX does underneath)
  <https://developer.nvidia.com/blog/enabling-multi-node-nvlink-on-kubernetes-for-gb200-and-beyond/>

---

## Layer above — how serving rides all of this

- **NCCL environment variables** (MNNVL selection, e.g. `NCCL_MNNVL_ENABLE`, and
  NVLink transport control) — under
  <https://docs.nvidia.com/deeplearning/nccl/>
- **SGLang — PD Disaggregation** (prefill/decode roles, Mooncake/NIXL transfer
  backends, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`)
  <https://docs.sglang.ai/advanced_features/pd_disaggregation.html>
- **Mooncake — Transfer Engine** (transport classes incl. NVLink/MNNVL,
  `MC_FORCE_MNNVL`, topology-aware path selection)
  <https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html>
- **NVIDIA Dynamo — SGLang disaggregation** (orchestration on top of NIXL/Mooncake)
  <https://docs.nvidia.com/dynamo/latest/backends/sglang/sglang-disaggregation.html>

---

## Fastest path to a working mental model

If you only read two things: the **IMEX guide — Overview** (Layer 1) and the
**DGX GB "Software" chapter** (Layer 2). Together they give ~80% of the model;
everything else is depth on one of the boxes they describe.

---

*Note: this stack evolves quickly. Where a page shows a version in its URL
(e.g. Mission Control 2.0.0), check for a newer revision. Serving-layer flags
(SGLang/Mooncake) in particular change release-to-release — verify against the
version bundled in your container image.*

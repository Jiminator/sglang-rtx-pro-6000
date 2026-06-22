# DeepSeek-V4-Pro FP8 — not yet benchmarked on this cluster

Placeholder. The `dsv4-8k64k` campaign tuned **DeepSeek-V4-Flash** (see
[`../../flash/fp8/`](../../flash/fp8/)); DeepSeek-V4-Pro FP8 has not been benchmarked on the
RTX PRO 6000 (SM120) node yet.

Expected starting point when run: the same SM120 backend constraints apply
(`--moe-runner-backend triton`, `--moe-a2a-backend none`, KV auto-locks fp8_e4m3), and
**DP-attention** should again be the lever if Pro's decode is attention-bound. Begin from the Flash
FP8 launch recipe and re-profile before assuming the bottleneck.

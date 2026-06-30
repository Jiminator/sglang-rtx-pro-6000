#!/usr/bin/env python3
"""GLM-5.2-NVFP4 serving Pareto plots — throughput/GPU vs per-user tok/s.
y-axis: output token throughput per GPU (tok/s/GPU)
x-axis: per-user token rate (tok/s/user = 1000 / mean ITL)
Each marker is one concurrency level; up-and-to-the-right is better.
Data: sglang.bench_serving, single node (8x RTX PRO 6000 SM120), ISL 1024 / OSL 8192, glm-opt build.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEC = "#2E6FF2"   # EAGLE (speculative)
BASE = "#E08A1E"   # non-spec
HC   = "#1FA88F"   # +HiCache
GRID = "#D9DEE6"

def tsu(itl):  # tokens/s/user from mean ITL (ms)
    return [1000.0/x for x in itl]

# ---- Random 1k/8k (no cache), cc 1..128 ----
R_CC   = [1,2,4,8,16,32,64,128]
NS_T   = [1.95,3.81,7.39,14.31,25.75,46.18,81.62,141.0]
NS_ITL = [64.16,65.43,67.58,69.71,77.41,86.26,97.22,112.0]
EA_T   = [6.83,13.17,23.68,45.02,76.13,127.45,199.79,277.97]
EA_ITL = [18.24,18.90,19.98,20.84,24.10,28.36,34.35,48.47]

# ---- Zipfian shared-prefix, cc 1..512 ----
Z_CC    = [1,2,4,8,16,32,64,128,256,512]
ZNS_T   = [2.37,4.62,8.13,17.54,31.21,56.17,98.92,132.73,145.25,166.34]
ZNS_ITL = [52.76,54.04,55.40,56.86,63.84,70.90,80.31,91.88,136.52,160.05]
ZHC_T   = [2.36,4.60,8.10,17.53,31.58,57.52,99.30,133.45,147.15,171.09]
ZHC_ITL = [52.88,54.27,55.61,56.87,63.05,69.20,79.97,92.10,135.59,157.41]
ZEA_T   = [8.1,14.69,27.1,53.79,90.08,157.2,142.03,128.75,125.93,121.31]
ZEA_ITL = [15.31,16.58,16.82,16.98,19.98,23.84,42.16,73.75,124.46,140.68]

def style(ax, title, sub):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#8A93A0")
    ax.set_xlabel("per-user throughput  (tok/s/user = 1000 / ITL)  → faster", fontsize=10.5)
    ax.set_ylabel("output throughput per GPU  (tok/s/GPU)  → cheaper", fontsize=10.5)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color="#5C6573")
    ax.tick_params(colors="#5C6573", labelsize=9)

def curve(ax, x, y, cc, color, label, marker="o", ls="-", lbl_idx=None):
    ax.plot(x, y, ls=ls, color=color, lw=2.2, marker=marker, ms=6.5, mfc=color,
            mec="white", mew=1.0, label=label, zorder=4)
    if lbl_idx is None: lbl_idx = range(len(cc))
    for i in lbl_idx:
        ax.annotate(f"cc{cc[i]}", (x[i], y[i]), textcoords="offset points",
                    xytext=(6,5), fontsize=7.5, color=color, fontweight="bold")

# ===== Figure 1: random 1k/8k — non-spec vs EAGLE =====
fig, ax = plt.subplots(figsize=(8.2,5.6), dpi=150)
style(ax, "GLM-5.2-NVFP4 — throughput / latency Pareto (random 1k/8k)",
      "1 node · 8× RTX PRO 6000 SM120 · fp8 KV + DP-attention · bench_serving")
curve(ax, tsu(NS_ITL), NS_T, R_CC, BASE, "non-spec (fp8 KV)", marker="s")
curve(ax, tsu(EA_ITL), EA_T, R_CC, SPEC, "EAGLE 3-step (spec, accept-len 3.9)", marker="o")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.text(0.99, 0.02, "EAGLE Pareto-dominates: higher tok/s/GPU AND faster per user at every concurrency",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#5C6573", style="italic")
fig.tight_layout()
fig.savefig("pareto_random_1k8k.png", bbox_inches="tight", facecolor="white")
print("wrote pareto_random_1k8k.png")

# ===== Figure 2: zipfian shared-prefix — radix vs +HiCache vs EAGLE =====
fig, ax = plt.subplots(figsize=(8.2,5.6), dpi=150)
style(ax, "GLM-5.2-NVFP4 — Pareto on zipfian shared-prefix (radix vs +HiCache)",
      "1 node · SM120 · zipf α=1.1 · mfs0.92 · cc 1→512 · bench_serving")
lab = [0,3,5,7,9]  # label cc 1,8,32,128,512
curve(ax, tsu(ZNS_ITL), ZNS_T, Z_CC, BASE, "non-spec · radix L1 only", marker="s", lbl_idx=lab)
curve(ax, tsu(ZHC_ITL), ZHC_T, Z_CC, HC,   "non-spec · + L2 HiCache", marker="^", ls="--", lbl_idx=[9])
curve(ax, tsu(ZEA_ITL), ZEA_T, Z_CC, SPEC, "EAGLE 3-step · radix L1", marker="o", lbl_idx=lab)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.text(0.01, 0.02, "L2 HiCache adds ~3% over radix-L1; EAGLE peaks mid-cc then queue-degrades at high load",
        transform=ax.transAxes, ha="left", fontsize=8.5, color="#5C6573", style="italic")
fig.tight_layout()
fig.savefig("pareto_zipfian_1k8k.png", bbox_inches="tight", facecolor="white")
print("wrote pareto_zipfian_1k8k.png")

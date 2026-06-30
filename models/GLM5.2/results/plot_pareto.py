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
GRID = "#D9DEE6"

def tsu(itl):  # tokens/s/user from mean ITL (ms)
    return [1000.0/x for x in itl]

# ---- Random 1k/8k (no cache), cc 1..128 ----
R_CC   = [1,2,4,8,16,32,64,128]
NS_T   = [1.95,3.81,7.39,14.31,25.75,46.18,81.62,141.0]
NS_ITL = [64.16,65.43,67.58,69.71,77.41,86.26,97.22,112.0]
EA_T   = [6.83,13.17,23.68,45.02,76.13,127.45,199.79,277.97]
EA_ITL = [18.24,18.90,19.98,20.84,24.10,28.36,34.35,48.47]

# ---- Zipfian shared-prefix, cc 1..512 (4 curves: {non-spec, EAGLE} x {radix-L1, +L2 HiCache}) ----
Z_CC     = [1,2,4,8,16,32,64,128,256,512]
ZNS_T    = [2.37,4.62,8.13,17.54,31.21,56.17,98.92,132.73,145.25,166.34]
ZNS_ITL  = [52.76,54.04,55.40,56.86,63.84,70.90,80.31,91.88,136.52,160.05]
ZNS_HIT  = [55.8,14.0,28.2,28.0,31.5,42.0,44.2,46.0,57.1,60.1]
ZHC_T    = [2.36,4.60,8.10,17.53,31.58,57.52,99.30,133.45,147.15,171.09]
ZHC_ITL  = [52.88,54.27,55.61,56.87,63.05,69.20,79.97,92.10,135.59,157.41]
ZHC_HIT  = [0.0,14.07,28.04,28.07,31.21,42.0,44.21,46.93,64.41,65.87]
ZEA_T    = [8.1,14.69,27.1,53.79,90.08,157.2,142.03,128.75,125.93,121.31]
ZEA_ITL  = [15.31,16.58,16.82,16.98,19.98,23.84,42.16,73.75,124.46,140.68]
ZEA_HIT  = [0.0,14.0,28.2,28.1,31.4,36.0,35.5,37.4,36.2,36.6]
ZEAH_T   = [7.97,15.91,27.16,56.09,99.57,132.18,141.06,137.86,125.50,127.33]
ZEAH_ITL = [15.57,15.58,16.50,16.97,19.09,25.26,41.06,67.49,123.67,134.53]
ZEAH_HIT = [0.0,13.98,27.96,27.89,31.49,42.16,43.07,42.58,55.66,54.83]

def style(ax, title, sub, xlabel, ylabel):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#8A93A0")
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color="#5C6573")
    ax.tick_params(colors="#5C6573", labelsize=9)

def curve(ax, x, y, cc, color, label, marker="o", ls="-", lbl_idx=None):
    ax.plot(x, y, ls=ls, color=color, lw=2.0, marker=marker, ms=6.0, mfc=color,
            mec="white", mew=1.0, label=label, zorder=4)
    if lbl_idx is None: lbl_idx = range(len(cc))
    for i in lbl_idx:
        ax.annotate(f"cc{cc[i]}", (x[i], y[i]), textcoords="offset points",
                    xytext=(6,5), fontsize=7.5, color=color, fontweight="bold")

XL = "per-user throughput  (tok/s/user = 1000 / ITL)  → faster"
YL = "output throughput per GPU  (tok/s/GPU)  → cheaper"

# ===== Figure 1: random 1k/8k — non-spec vs EAGLE =====
fig, ax = plt.subplots(figsize=(8.2,5.6), dpi=150)
style(ax, "GLM-5.2-NVFP4 — throughput / latency Pareto (random 1k/8k)",
      "1 node · 8× RTX PRO 6000 SM120 · fp8 KV + DP-attention · bench_serving", XL, YL)
curve(ax, tsu(NS_ITL), NS_T, R_CC, BASE, "non-spec (fp8 KV)", marker="s")
curve(ax, tsu(EA_ITL), EA_T, R_CC, SPEC, "EAGLE 3-step (spec, accept-len 3.9)", marker="o")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.text(0.99, 0.02, "EAGLE Pareto-dominates: higher tok/s/GPU AND faster per user at every concurrency",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#5C6573", style="italic")
fig.tight_layout(); fig.savefig("pareto_random_1k8k.png", bbox_inches="tight", facecolor="white")
print("wrote pareto_random_1k8k.png")

# ===== Figure 2: zipfian shared-prefix — 4 curves (paired by config color, cache = solid/dashed) =====
fig, ax = plt.subplots(figsize=(8.2,5.6), dpi=150)
style(ax, "GLM-5.2-NVFP4 — Pareto on zipfian shared-prefix (radix L1 vs + L2 HiCache)",
      "1 node · SM120 · zipf α=1.1 · mfs0.92 · cc 1→512 · solid = radix L1, dashed = + HiCache", XL, YL)
lab = [0,5,9]  # label cc 1, 32, 512
curve(ax, tsu(ZNS_ITL),  ZNS_T,  Z_CC, BASE, "non-spec · radix L1",   marker="s", ls="-",  lbl_idx=lab)
curve(ax, tsu(ZHC_ITL),  ZHC_T,  Z_CC, BASE, "non-spec · + HiCache",  marker="^", ls="--", lbl_idx=[])
curve(ax, tsu(ZEA_ITL),  ZEA_T,  Z_CC, SPEC, "EAGLE · radix L1",      marker="o", ls="-",  lbl_idx=lab)
curve(ax, tsu(ZEAH_ITL), ZEAH_T, Z_CC, SPEC, "EAGLE · + HiCache",     marker="D", ls="--", lbl_idx=[])
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.text(0.01, 0.02, "HiCache barely shifts throughput (decode-bound); its win is hit-rate — see the next panel",
        transform=ax.transAxes, ha="left", fontsize=8.5, color="#5C6573", style="italic")
fig.tight_layout(); fig.savefig("pareto_zipfian_1k8k.png", bbox_inches="tight", facecolor="white")
print("wrote pareto_zipfian_1k8k.png")

# ===== Figure 3: zipfian cache hit-rate vs concurrency (the actual HiCache effect) =====
fig, ax = plt.subplots(figsize=(8.2,5.0), dpi=150)
style(ax, "GLM-5.2-NVFP4 — zipfian prefix cache hit-rate vs concurrency",
      "L2 HiCache (dashed) keeps the hit-rate up where the L1 radix tree evicts under load",
      "client concurrency (log)", "prefix cache hit-rate  (%)")
ax.set_xscale("log", base=2)
ax.plot(Z_CC, ZNS_HIT,  color=BASE, ls="-",  marker="s", lw=2.0, ms=6, mec="white", label="non-spec · radix L1")
ax.plot(Z_CC, ZHC_HIT,  color=BASE, ls="--", marker="^", lw=2.0, ms=6, mec="white", label="non-spec · + HiCache")
ax.plot(Z_CC, ZEA_HIT,  color=SPEC, ls="-",  marker="o", lw=2.0, ms=6, mec="white", label="EAGLE · radix L1")
ax.plot(Z_CC, ZEAH_HIT, color=SPEC, ls="--", marker="D", lw=2.0, ms=6, mec="white", label="EAGLE · + HiCache")
ax.set_xticks(Z_CC); ax.set_xticklabels([str(c) for c in Z_CC])
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.annotate("L2 lifts EAGLE +19 pts", (256, 55.66), textcoords="offset points", xytext=(-8,12),
            fontsize=8.5, color=SPEC, fontweight="bold")
fig.tight_layout(); fig.savefig("hitrate_zipfian_1k8k.png", bbox_inches="tight", facecolor="white")
print("wrote hitrate_zipfian_1k8k.png")

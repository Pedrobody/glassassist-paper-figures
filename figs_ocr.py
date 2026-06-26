import csv, sys, glob, os, json, statistics as st
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(7)
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"

rows = list(csv.DictReader(open(OUT + r"\bench_ocr.csv")))
lines = np.array([int(r["line_count"]) for r in rows])
gpu = np.array([float(r["gpu_ms"]) for r in rows])
cpu = np.array([float(r["cpu_ms"]) for r in rows])
TTL = 6000      # belief freshness window
TIMEOUT = 8000  # Glass OkHttp read timeout

# ---- real ML Kit (on-device) latency median, pooled over all "before" sessions ----
def mlkit_latencies():
    vals = []
    for d in glob.glob(r"C:\Users\Pedro\pc_server_sessions\*\events.jsonl"):
        for line in open(d, encoding="utf-8"):
            try: e = json.loads(line)
            except: continue
            if e.get("type") == "ocr.result":
                p = e.get("payload") or {}
                if p.get("source") != "pc" and p.get("latency_ms") is not None:
                    vals.append(p["latency_ms"])
    return vals
mlkit = mlkit_latencies()
mlkit_med = st.median(mlkit) if mlkit else 1100
print(f"ML Kit on-device: n={len(mlkit)} median={mlkit_med:.0f}ms")

# ---------- FIG 1: distribution CPU vs GPU ----------
fig, ax = plt.subplots(figsize=(6.0, 4.3))
data = [cpu, gpu]
labels = ["PC · CPU\n(isolated, 4 threads)", "PC · GPU\n(RTX 3080 Ti)"]
colors = [cava.BEFORE, cava.AFTER]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color=cava.NAVY, lw=2.2),
                whiskerprops=dict(color=cava.NAVY), capprops=dict(color=cava.NAVY))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(.22); patch.set_edgecolor(c); patch.set_linewidth(1.6)
for i, (d, c) in enumerate(zip(data, colors), start=1):
    x = np.random.normal(i, 0.055, len(d))
    ax.scatter(x, d, s=20, color=c, alpha=.75, zorder=3, edgecolor="white", linewidth=.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
ax.set_ylabel("On-server OCR latency per frame (ms)")
ax.set_title("OCR compute time: CPU → GPU  (same 25 frames)")
ax.text(1, np.median(cpu), f"  median {np.median(cpu):.0f} ms", va="center", ha="left", fontsize=9, color=cava.NAVY, fontweight="bold")
ax.text(2, np.median(gpu), f"  median {np.median(gpu):.0f} ms", va="center", ha="left", fontsize=9, color=cava.NAVY, fontweight="bold")
ax.annotate(f"{np.median(cpu)/np.median(gpu):.1f}× faster", xy=(2, np.median(gpu)), xytext=(1.5, 3200),
            ha="center", fontsize=10, fontweight="bold", color=cava.GREEN,
            arrowprops=dict(arrowstyle="->", color=cava.GREEN, lw=1.4))
ax.set_ylim(0, max(cpu)*1.1)
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_ocr_latency_box.png", bbox_inches="tight"); print("saved fig_ocr_latency_box.png")

# ---------- FIG 2: latency vs scene clutter ----------
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ymax = 6800
ax.axhspan(TTL, ymax, color=cava.ORANGE, alpha=.07)
ax.axhline(TTL, color=cava.ORANGE, ls="--", lw=1.3)
ax.text(lines.max(), TTL+170, "6 s freshness window — above = stale (dropped)",
        color=cava.ORANGE, fontsize=8.6, fontweight="bold", va="bottom", ha="right")
ax.scatter(lines, cpu, s=44, color=cava.BEFORE, alpha=.8, edgecolor="white", linewidth=.5, label="PC · CPU (isolated)", zorder=3)
ax.scatter(lines, gpu, s=44, color=cava.AFTER, alpha=.9, edgecolor="white", linewidth=.5, label="PC · GPU", zorder=3)
for d, c in [(cpu, cava.BEFORE), (gpu, cava.AFTER)]:
    z = np.polyfit(lines, d, 1); xs = np.array([0, lines.max()])
    ax.plot(xs, np.polyval(z, xs), color=c, lw=1.5, alpha=.45)
ax.set_xlabel("Scene clutter — number of text regions in frame")
ax.set_ylabel("OCR latency (ms)")
ax.set_title("CPU latency scales with clutter; GPU stays bounded")
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62))
ax.set_ylim(0, ymax); ax.set_xlim(-1.5, lines.max()+3)
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_ocr_vs_clutter.png", bbox_inches="tight"); print("saved fig_ocr_vs_clutter.png")

# ---------- FIG 3: the optimization journey ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars = [
    ("On-device\nML Kit\n(hot Glass)", mlkit_med, cava.GREY_D, "baseline being\nreplaced"),
    ("PC · CPU\nnaïve\n(in torch proc)", 8500, cava.ORANGE, "5–47 s · OpenMP\ncontention → 20% lost"),
    ("PC · CPU\nisolated\n(4 threads)", np.median(cpu), cava.TEAL, f"{np.median(cpu):.0f} ms median"),
    ("PC · GPU\n(RTX)", np.median(gpu), cava.GREEN, f"{np.median(gpu):.0f} ms median · 0% lost"),
]
xs = np.arange(len(bars))
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]
ax.bar(xs, vals, width=0.62, color=cols, alpha=.85, edgecolor=cava.NAVY, linewidth=.6, zorder=3)
ax.set_yscale("log")
ax.axhline(TIMEOUT, color=cava.ORANGE, ls="--", lw=1.4, zorder=2)
ax.text(len(bars)-0.5, TIMEOUT*1.06, "8 s Glass timeout", color=cava.ORANGE, ha="right", fontsize=9, fontweight="bold")
ax.axhline(TTL, color=cava.GREY_D, ls=":", lw=1.1, zorder=2)
ax.text(len(bars)-0.5, TTL*0.80, "6 s freshness", color=cava.GREY_D, ha="right", fontsize=8)
for x, b in zip(xs, bars):
    ax.text(x, b[1]*1.25, b[3], ha="center", va="bottom", fontsize=8, color=cava.NAVY)
ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_ylabel("OCR latency per frame (ms, log scale)")
ax.set_title("Making OCR offload viable: from naïve CPU to GPU")
ax.set_ylim(200, 60000)
ax.grid(axis="x", visible=False)
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_ocr_journey.png", bbox_inches="tight"); print("saved fig_ocr_journey.png")

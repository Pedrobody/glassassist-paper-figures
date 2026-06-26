import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"
THR = 0.60
OCR_C = 0.25
SEC_C = 0.20

ocr = np.linspace(0, 1, 200)
def before(owl, vlm, o):  # noisy-OR with OCR inside
    return 1 - (1-owl)*(1-vlm)*(1-o)
def after(owl, vlm, o):   # OCR as bounded corroboration
    return np.clip((1-(1-owl)*(1-vlm)) + OCR_C*o, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
levels = [(0.0, cava.ORANGE, "OWL=VLM= 0  (OCR alone)"),
          (0.3, cava.TEAL,   "OWL= 0.3"),
          (0.5, cava.GREEN,  "OWL= 0.5")]
for ax, fn, title, tag in [(axes[0], before, "BEFORE — OCR in the noisy-OR", "ocr=1 → TargetSeen=1 (fires alone)"),
                           (axes[1], after,  "AFTER — OCR is bounded corroboration", "ocr=1 → +0.25 (never fires alone)")]:
    ax.axhspan(THR, 1.02, color=cava.GREEN, alpha=.05)
    ax.axhline(THR, color=cava.NAVY, ls="--", lw=1.3)
    ax.text(0.02, THR+0.015, "TargetSeen threshold 0.60", color=cava.NAVY, fontsize=8.5, fontweight="bold")
    for lv, c, lab in levels:
        ax.plot(ocr, fn(lv, 0.0, ocr), color=c, lw=2.4, label=lab)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("OCR match score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    # marker for OCR-alone at ocr=1
    y1 = fn(0.0, 0.0, 1.0)
    ax.scatter([1], [y1], s=70, color=cava.ORANGE, zorder=5, edgecolor=cava.NAVY, linewidth=.8)
    ax.annotate(tag, xy=(1, y1), xytext=(0.42, y1+ (0.12 if y1<0.6 else -0.16)),
                fontsize=9, color=cava.ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=cava.ORANGE, lw=1.3))
axes[0].set_ylabel("TargetSeen score")
axes[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("Evidence fusion: making OCR a supporting signal, not a standalone trigger", fontweight="bold")
cava.footer(fig); fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT + r"\fig_fusion_before_after.png", bbox_inches="tight"); print("saved fig_fusion_before_after.png")

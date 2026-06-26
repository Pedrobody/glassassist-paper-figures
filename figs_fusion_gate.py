import sys, json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"

GATE = 0.25; THR = 0.60; OCR_C = 0.25; SEC_C = 0.20

# --- decision plane in (OWL, VLM) space -------------------------------------
owl = np.linspace(0, 1, 500); vlm = np.linspace(0, 1, 500)
OWL, VLM = np.meshgrid(owl, vlm)
direct = 1 - (1 - OWL) * (1 - VLM)          # noisy-OR of the two DIRECT detectors

# zone: 0 = gated (corroboration ignored), 1 = corroboration-assisted, 2 = detectors fire alone
zone = np.zeros_like(direct)
zone[(direct >= GATE) & (direct < THR)] = 1
zone[direct >= THR] = 2

fig, ax = plt.subplots(figsize=(8.4, 7.2))
ax.contourf(OWL, VLM, zone, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=["#FBE3D2", "#DDEBF1", "#E2F0E3"])           # light orange / teal / green

# boundaries
c1 = ax.contour(OWL, VLM, direct, levels=[GATE], colors=[cava.ORANGE], linewidths=2.4, linestyles="--")
c2 = ax.contour(OWL, VLM, direct, levels=[THR],  colors=[cava.GREEN],  linewidths=2.4)
ax.clabel(c1, fmt={GATE: "gate  direct = 0.25"}, fontsize=8.5, inline_spacing=6)
ax.clabel(c2, fmt={THR:  "fire  direct = 0.60"}, fontsize=8.5, inline_spacing=6)

# zone captions
ax.annotate("GATED\ncorroboration ignored\n(never fires on text alone)",
            xy=(0.04, 0.12), xytext=(0.30, 0.07),
            fontsize=9.0, color=cava.ORANGE, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=cava.ORANGE, lw=1.0),
            arrowprops=dict(arrowstyle="->", color=cava.ORANGE, lw=1.2))
ax.text(0.54, 0.42, "CORROBORATION ZONE\nOCR (+0.25·s) + section (+0.20·s)\nlift a real-but-weak detection over 0.60",
        fontsize=9.0, color=cava.TEAL, fontweight="bold", ha="center", va="center")
ax.text(0.62, 0.72, "DETECTORS FIRE ALONE\nOWL / VLM strong (direct ≥ 0.60)",
        fontsize=9.0, color=cava.GREEN, fontweight="bold", ha="center", va="center")

# --- real data points from the labelled "white wine" session ----------------
SD = r"C:\Users\Pedro\pc_server_sessions"
sess = sorted([d for d in glob.glob(os.path.join(SD, "*")) if os.path.isdir(d)
               and "lat_test" not in d], key=os.path.getmtime)[-1]
pts = []
for line in open(os.path.join(sess, "events.jsonl"), encoding="utf-8"):
    try: e = json.loads(line)
    except: continue
    if e.get("type") == "targetseen.fusion":
        p = e.get("payload") or {}
        pts.append((p.get("object_score", 0), p.get("vlm_visible", 0), p.get("ocr_score", 0)))
pts = np.array(pts)
if len(pts):
    # tiny x-jitter so the OWL=0 column is visible as a cloud, not a line
    jx = pts[:, 0] + np.linspace(-0.006, 0.006, len(pts))
    ax.scatter(jx, pts[:, 1], s=14, color=cava.NAVY, alpha=0.30, zorder=4,
               label=f"observed frames (n={len(pts)})")
    # the two storytelling points
    ax.scatter([0.0], [0.18], s=170, marker="X", color=cava.ORANGE, edgecolor="white",
               linewidth=1.2, zorder=6)
    ax.annotate("text-only false fires\n(OWL=0, VLM≈0.18, fired by OCR+section)",
                xy=(0.0, 0.18), xytext=(0.34, 0.20), fontsize=8.4, color=cava.ORANGE, fontweight="bold",
                va="center", arrowprops=dict(arrowstyle="->", color=cava.ORANGE, lw=1.3))
    ax.scatter([0.0], [0.28], s=230, marker="*", color=cava.GREEN, edgecolor="white",
               linewidth=1.2, zorder=6)
    ax.annotate("real wine: weak VLM (0.28) + OCR\n→ now fires as intended",
                xy=(0.0, 0.28), xytext=(0.10, 0.50), fontsize=8.4, color=cava.GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=cava.GREEN, lw=1.3))

ax.set_xlim(0, 0.82); ax.set_ylim(0, 0.82)
ax.set_xlabel(r"OWL open-vocab object score  $s_{\mathrm{OWL}}$")
ax.set_ylabel(r"VLM visibility score  $s_{\mathrm{VLM}}$")
ax.set_title("Gated corroboration — OCR/section refine, never originate, a detection")
ax.legend(loc="upper right")
fig.text(0.5, -0.01,
         r"$\mathrm{direct}=1-(1-s_{OWL})(1-s_{VLM})$;  "
         r"corroboration $=\mathbb{1}[\mathrm{direct}\geq 0.25]\,(0.25\,s_{OCR}+0.20\,s_{SEC})$;  "
         r"fires if $\mathrm{direct}+\mathrm{corroboration}\geq 0.60$.",
         ha="center", fontsize=8.4, color=cava.GREY_D)
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_fusion_gate.png", bbox_inches="tight")
print("saved fig_fusion_gate.png from session", os.path.basename(sess))

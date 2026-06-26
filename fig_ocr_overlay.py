import sys, glob, os, json, urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"

def ocr(path):
    raw = open(path, "rb").read()
    req = urllib.request.Request("http://127.0.0.1:8090/ocr?lang=en", data=raw, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode()).get("lines", [])

# Fixed, representative shelf frame (pasta / sauce aisle, many recognizable products).
CONF_MIN = 0.72
best = r"C:\Users\Pedro\pc_server_sessions\20260625-170148_750892\frames\000753.jpg"
best_lines = [l for l in ocr(best) if l["conf"] >= CONF_MIN and len(l["text"].strip()) >= 3]
print("rendering", os.path.basename(best), "with", len(best_lines), "confident lines (>=%.2f)" % CONF_MIN)

# --- render overlay ---
img = cv2.cvtColor(cv2.imread(best), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
fig, ax = plt.subplots(figsize=(9.0, 9.0*h/w + 0.6))
ax.imshow(img)
ax.axis("off")
for l in best_lines:
    b = l["box"]; conf = l["conf"]; txt = l["text"].strip()
    x, y, bw, bh = b["x"]*w, b["y"]*h, b["w"]*w, b["h"]*h
    c = cava.GREEN if conf >= 0.85 else cava.TEAL
    ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=c, lw=2.2))
    ax.text(x, max(y-4, 9), f"{txt}  {conf:.2f}", fontsize=7.6, color="white", fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", fc=c, ec="none", alpha=.93))
ax.set_title("On-server OCR — recognized text & confidence on a real shelf frame",
             fontsize=12, color=cava.NAVY, fontweight="bold")
fig.text(0.5, 0.01, "Confident reads only (conf ≥ 0.72). Boxes are returned normalized by the GPU OCR service (~0.5 s/frame) and re-projected here.",
         ha="center", fontsize=8, color=cava.GREY_D, style="italic")
fig.tight_layout()
fig.savefig(OUT + r"\fig_ocr_overlay.png", bbox_inches="tight", dpi=190)
print("saved fig_ocr_overlay.png")

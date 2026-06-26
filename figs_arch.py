import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"
N = cava.NAVY; G = cava.GREEN; O = cava.ORANGE; T = cava.TEAL; GD = cava.GREY_D; GREY = cava.GREY

def box(ax, x, y, w, h, text, fc, ec=N, tc=N, fs=9.5, lw=1.3, sub=None, subc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if sub:
        ax.text(x+w/2, y+h*0.63, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=3)
        ax.text(x+w/2, y+h*0.27, sub, ha="center", va="center", fontsize=fs-2.0, color=subc or GD, zorder=3)
    else:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=3)

def arr(ax, p1, p2, color=N, lw=1.7, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color,
                                 lw=lw, zorder=1, shrinkA=3, shrinkB=3))

def draw(mode):
    after = (mode == "after")
    fig, ax = plt.subplots(figsize=(11.2, 5.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    # ---- Glass container ----
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 4.8, 5.25, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#F4F7F6", ec=N, lw=1.8, zorder=0))
    ax.text(2.65, 5.42, "Google Glass EE2  ·  Snapdragon XR1 (4 cores)", ha="center", fontsize=10, fontweight="bold", color=N)
    badge = ("OCR offloaded + 8 fps  →  cooler" if after else "ML Kit OCR + 15 fps  →  heat / throttling")
    ax.text(2.65, 0.18, badge, ha="center", fontsize=8.4, fontweight="bold",
            color=(G if after else O))

    box(ax, 0.85, 4.62, 3.6, 0.55, "Camera (CameraX)", "white")
    box(ax, 0.85, 3.82, 3.6, 0.6, "YUV → RGB convert", "white",
        sub=("8 fps (capped)" if after else "15 fps"), subc=(G if after else GD))
    # 3 detectors
    dy, dh = 2.78, 0.66
    box(ax, 0.78, dy, 1.12, dh, "OWL", "white", fs=9)
    box(ax, 2.04, dy, 1.12, dh, "VLM", "white", fs=9)
    ocr_fc = ("white" if after else "#FBE3D2")
    box(ax, 3.30, dy, 1.12, dh, "OCR", ocr_fc, ec=(N if after else O),
        sub=("upload" if after else "ML Kit·CPU"), subc=(G if after else O), fs=9)
    box(ax, 0.85, 1.86, 3.6, 0.62, "TargetSeen fusion", "white",
        sub=("OWL,VLM ⊕ OCR/section (capped)" if after else "noisy-OR: OWL, VLM, OCR"),
        subc=(G if after else GD))
    box(ax, 1.35, 1.08, 2.6, 0.5, "Voice guidance", "white", fs=9)

    # internal arrows
    arr(ax, (2.65, 4.62), (2.65, 4.42))            # camera -> convert
    arr(ax, (2.65, 3.82), (2.65, 3.45))            # convert -> detectors band
    for cx in (1.34, 2.60, 3.86):
        arr(ax, (cx, dy), (2.65, 2.48), color=GD, lw=1.1)   # detectors -> fusion
    arr(ax, (2.65, 1.86), (2.65, 1.58))            # fusion -> guidance

    # ---- PC server ----
    ax.add_patch(FancyBboxPatch((6.7, 3.05), 5.0, 2.45, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="white", ec=N, lw=1.8, zorder=0))
    ax.text(9.2, 5.28, "PC server (LAN)", ha="center", fontsize=10, fontweight="bold", color=N)
    box(ax, 7.0, 4.35, 4.4, 0.62, ("OWL: OWLv2-B" if after else "OWL: Grounding DINO"), "#E7F0E8" if after else "white",
        ec=G if after else N, sub="GPU (RTX 3080 Ti)", subc=G if after else GD)
    if after:
        box(ax, 7.0, 3.18, 2.05, 0.95, "/ocr/infer\nproxy", "white", fs=8.5)
        box(ax, 9.35, 3.18, 2.05, 0.95, "OCR process\n(isolated)", "#E7F0E8", ec=G, fs=8.5, sub="GPU", subc=G)
        arr(ax, (9.05, 3.65), (9.35, 3.65), color=G)
    else:
        ax.text(9.2, 3.55, "(OCR not here — runs on the Glass)", ha="center", fontsize=8, style="italic", color=GD)

    # ---- Cloud ----
    ax.add_patch(FancyBboxPatch((6.7, 0.7), 5.0, 1.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="white", ec=N, lw=1.8, zorder=0))
    ax.text(9.2, 2.18, "OpenAI cloud", ha="center", fontsize=10, fontweight="bold", color=N)
    box(ax, 7.0, 1.05, 4.4, 0.78, "VLM scene understanding", "white", fs=9)

    # ---- cross arrows (detector -> backend) ----
    arr(ax, (1.90, 3.11), (7.0, 4.62), color=T, lw=1.8)        # OWL -> PC
    ax.text(4.6, 4.15, "OWL frame", fontsize=7.6, color=T, rotation=12, fontweight="bold")
    arr(ax, (3.16, 3.0), (7.0, 1.5), color=T, lw=1.8)          # VLM -> cloud
    ax.text(4.7, 2.05, "VLM frame", fontsize=7.6, color=T, rotation=-20, fontweight="bold")
    if after:
        arr(ax, (4.42, 3.0), (7.0, 3.55), color=G, lw=2.0)     # OCR -> PC
        ax.text(5.0, 3.42, "OCR frame", fontsize=7.6, color=G, rotation=6, fontweight="bold")

    ax.set_title(("AFTER — OCR runs on the PC GPU; capture capped to 8 fps; OWLv2-B; OCR = corroboration"
                  if after else
                  "BEFORE — OCR runs on-device (ML Kit); 15 fps; OCR fires in the noisy-OR"),
                 fontsize=11.5, color=N, fontweight="bold", pad=14)
    cava.footer(fig); fig.tight_layout()
    f = OUT + (r"\fig_arch_after.png" if after else r"\fig_arch_before.png")
    fig.savefig(f, bbox_inches="tight"); print("saved", f.split("\\")[-1])

draw("before")
draw("after")

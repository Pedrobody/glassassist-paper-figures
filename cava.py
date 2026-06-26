"""CAVA / Aoyama Gakuin lab visual identity for paper-quality figures.
Navy + emerald green on white, with teal and an orange reserved for 'before/problem'.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette (from the lab DESIGN_MANUAL.md)
NAVY    = "#0E2841"   # dark base, text, axes
TEAL    = "#156082"   # secondary
GREEN   = "#196B24"   # primary "after / solution"
GREEN_L = "#4EA72E"   # light green accent
ORANGE  = "#E97132"   # "before / problem / warning"
GREY    = "#E8E8E8"   # light fill
GREY_D  = "#8A8A8A"   # muted

# Semantic roles for before/after storytelling
BEFORE  = ORANGE      # ML Kit / CPU / problem
MIDDLE  = TEAL        # transitional
AFTER   = GREEN       # GPU / solution

def apply():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GREY,
        "grid.linewidth": 0.8,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "font.family": "DejaVu Sans",   # Aptos-like clean sans (Aptos not bundled)
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })

def footer(fig, text="GlassAssist · CAVA Lab, Aoyama Gakuin University"):
    fig.text(0.99, 0.005, text, ha="right", va="bottom",
             fontsize=7, color=GREY_D, style="italic")

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"

# Measured via ADB (Snapdragon XR1, Glass EE2). Frequencies in kHz.
cores = ["cpu0", "cpu1", "cpu2", "cpu3"]
hwmax = np.array([1209600, 1209600, 1363200, 1363200]) / 1e6   # GHz (cpuinfo_max_freq)
cold  = np.array([1209600, 1209600, 1363200, 1363200]) / 1e6   # idle ~43C, cores reach max
hot   = np.array([748800,  748800,  825600,  825600])  / 1e6   # sustained load ~60C (8 reads, 0 variance)

x = np.arange(len(cores)); w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 4.5))
b1 = ax.bar(x - w/2, cold, w, color=cava.GREEN, alpha=.85, label="Cold (idle, ~43 °C)", edgecolor=cava.NAVY, linewidth=.5)
b2 = ax.bar(x + w/2, hot,  w, color=cava.ORANGE, alpha=.9, label="Hot (sustained load, ~60 °C)", edgecolor=cava.NAVY, linewidth=.5)
# hw max reference (== OS scaling_max_freq, which stays uncapped)
for i, m in enumerate(hwmax):
    ax.plot([i-0.5, i+0.5], [m, m], color=cava.NAVY, ls=":", lw=1.6, zorder=4)
ax.plot([], [], color=cava.NAVY, ls=":", lw=1.6, label="HW max = OS scaling_max_freq (uncapped)")

for i in range(len(cores)):
    pct = hot[i]/hwmax[i]*100
    ax.text(i + w/2, hot[i]-0.06, f"{pct:.0f}%", ha="center", va="top", color="white", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(cores)
ax.set_ylabel("Delivered CPU frequency (GHz)")
ax.set_title("Thermal throttle is in hardware — invisible to the OS")
ax.legend(loc="lower center", fontsize=8.6, ncol=1)
ax.set_ylim(0, 1.95)
ax.text(1.5, 1.62,
        "Under load at 60 °C the silicon delivers only ~61% of max,\n"
        "yet scaling_max_freq still reports 100% → no OS-level cap.\n"
        "The limiter is Qualcomm LMH-DCVS, below the cpufreq layer.",
        fontsize=8.8, color=cava.NAVY, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=cava.GREY_D, lw=.8))
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_freq_throttle.png", bbox_inches="tight"); print("saved fig_freq_throttle.png")

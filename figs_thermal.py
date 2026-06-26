import sys, json, glob, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Pedro\glass_diag\paper")
import cava
cava.apply()
OUT = r"C:\Users\Pedro\glass_diag\paper"
SD = r"C:\Users\Pedro\pc_server_sessions"

def owl_times(sess):
    ev = []
    for line in open(os.path.join(SD, sess, "events.jsonl"), encoding="utf-8"):
        try: e = json.loads(line)
        except: continue
        if e.get("type") == "owl.result" and e.get("t_server_ms"):
            ev.append(e["t_server_ms"])
    ev.sort()
    return np.array(ev)

def roll_med(x, k=5):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i-k+1)
        out[i] = np.median(x[lo:i+1])
    return out

# antepenult (degrades) then penult (quick reload, resets, re-degrades)
ante = owl_times("20260625-192015_61aa22")
pen  = owl_times("20260625-192432_bcb1ed")
t0 = ante[0]
def series(ev):
    t = (ev - t0) / 1000.0
    gap = np.diff(ev) / 1000.0
    return t[1:], gap
ta, ga = series(ante)
tp, gp = series(pen)

fig, ax = plt.subplots(figsize=(8.2, 4.4))
# raw (faint) + rolling median (bold)
ax.plot(ta, ga, color=cava.BEFORE, lw=0.8, alpha=.25)
ax.plot(tp, gp, color=cava.TEAL, lw=0.8, alpha=.25)
ax.plot(ta, roll_med(ga), color=cava.BEFORE, lw=2.6, label="Session A — sustained run")
ax.plot(tp, roll_med(gp), color=cava.TEAL, lw=2.6, label="Session B — quick reload (no cooldown)")

reload_t = (pen[0] - t0) / 1000.0
ax.axvline(reload_t, color=cava.NAVY, ls="--", lw=1.3, alpha=.7)
ax.text(reload_t, 3.75, "  app reloaded\n  (2 s gap, still hot)", color=cava.NAVY, fontsize=8.5, va="top", fontweight="bold")

ax.annotate("OWL slows 1 s → 3 s\nas the SoC heats", xy=(ta[-1], roll_med(ga)[-1]), xytext=(120, 3.4),
            fontsize=9, color=cava.BEFORE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=cava.BEFORE, lw=1.3))
ax.annotate("resets to ~1 s\n…then heats again", xy=(tp[2], roll_med(gp)[2]), xytext=(tp[2]+30, 0.55),
            fontsize=9, color=cava.TEAL, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=cava.TEAL, lw=1.3))

ax.set_xlabel("Time since start of Session A (s)")
ax.set_ylabel("OWL inter-execution interval (s)")
ax.set_title("The reload-recovery experiment: it's thermal, not software accumulation")
ax.legend(loc="upper left")
ax.set_ylim(0.5, 4.2)
fig.text(0.5, -0.02,
         "A 2 s app restart on a still-hot device fully restores speed — software state would persist, "
         "but the silicon junction cools in seconds.",
         ha="center", fontsize=8.5, color=cava.GREY_D, style="italic")
cava.footer(fig); fig.tight_layout()
fig.savefig(OUT + r"\fig_reload_recovery.png", bbox_inches="tight"); print("saved fig_reload_recovery.png")

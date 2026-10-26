"""Generate the figures used in the README. All data comes from the run logs."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
INK, GRID, ACC, MUT = "#10151b", "#dfe4ea", "#e0a013", "#6b7a89"
SER = ["#2a78d6", "#e0a013", "#1baf7a", "#d4537e", "#7f77dd"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

def tr(p):
    return [(r["iter"], r["test_psnr"], r["n"])
            for r in map(json.loads, open(p)) if "test_psnr" in r]

# ---- 1. the headline: what actually moved the needle ----------------------
items = [("SH schedule\n(1000 -> 300)", 0.56, SER[1]),
         ("depth-prior init\n(matched capacity)", 0.20, SER[0]),
         ("depth infill", 0.12, MUT),
         ("capacity x2", 0.07, MUT),
         ("surface-aligned\norientation", 0.06, MUT),
         ("densification\nknobs", 0.01, MUT),
         ("higher initial\nopacity", -0.51, "#b5352e")]
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(items))[::-1]
ax.barh(y, [i[1] for i in items], color=[i[2] for i in items], height=.62)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9)
for yy, (l, v, c) in zip(y, items):
    ax.text(v + (0.03 if v > 0 else -0.03), yy, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9, color=INK)
ax.axvline(0, color=MUT, lw=.8)
ax.set_xlabel("change in test PSNR (dB) at 2000 iterations")
ax.set_xlim(-0.72, 0.78); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("20 experiments. One hyperparameter beat every geometry prior.",
             fontsize=11.5, loc="left", pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_what_mattered.png", dpi=170); plt.close()

# ---- 2. the two signatures ------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
b = tr("logs/r2_m_random/train_log.jsonl"); d = tr("logs/r2_m_depth/train_log.jsonl")
it = [x[0] for x in b]
a1.plot(it, [x[1] for x in b], "o-", color=MUT, lw=2, ms=4, label="random init")
a1.plot(it, [x[1] for x in d], "o-", color=SER[0], lw=2, ms=4, label="depth-prior init")
a1.fill_between(it, [x[1] for x in b], [x[1] for x in d], color=SER[0], alpha=.12)
a1.set_title("Initialisation: gain DECAYS\n+2.06 dB at iter 250  ->  +0.20 at 2000",
             fontsize=10.5, loc="left")
bb = tr("logs/r3_flat10/train_log.jsonl"); s = tr("logs/r5_sh300/train_log.jsonl")
a2.plot(it, [x[1] for x in bb], "o-", color=MUT, lw=2, ms=4, label="sh_every=1000")
a2.plot(it, [x[1] for x in s], "o-", color=SER[1], lw=2, ms=4, label="sh_every=300")
a2.fill_between(it, [x[1] for x in bb], [x[1] for x in s], color=SER[1], alpha=.15)
a2.set_title("SH schedule: gain GROWS\n+0.00 dB at iter 250  ->  +0.56 at 2000",
             fontsize=10.5, loc="left")
for a in (a1, a2):
    a.set_xlabel("iteration"); a.set_ylabel("test PSNR (dB)")
    a.grid(True, color=GRID); a.set_axisbelow(True); a.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_signatures.png", dpi=170); plt.close()

# ---- 3. the hardware story ------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
lbl = ["fp32 matmul\n(2048^3)", "memory\nbandwidth", "3DGS trainer\n(the real workload)"]
gap = [32.6, 19.5, 5.8]
a1.bar(lbl, gap, color=[MUT, MUT, SER[1]], width=.55)
for i, v in enumerate(gap):
    a1.text(i, v + .8, f"{v:.1f}x", ha="center", fontsize=10.5, color=INK)
a1.set_ylabel("RTX 5080 advantage over Radeon 890M")
a1.set_ylim(0, 38); a1.yaxis.grid(True, color=GRID); a1.set_axisbelow(True)
a1.set_title("The spec sheet overstates the gap by 5x", fontsize=10.5, loc="left")
sizes = ["1024", "2048", "4096"]
x = np.arange(3); w = .26
for k, (nm, v, c) in enumerate([("fp32", [.73, .72, .68], SER[0]),
                                ("fp16", [3.83, 3.60, 3.23], SER[2]),
                                ("bf16", [2.81, 5.65, 4.02], SER[1])]):
    a2.bar(x + (k - 1) * w, v, w, label=nm, color=c)
a2.set_xticks(x); a2.set_xticklabels(sizes); a2.set_xlabel("matmul size")
a2.set_ylabel("TFLOP/s on the 890M"); a2.legend(frameon=False, fontsize=9)
a2.yaxis.grid(True, color=GRID); a2.set_axisbelow(True)
a2.set_title("bf16 gives the iGPU 7.8x  (the dGPU only 2.2x)", fontsize=10.5, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_hardware.png", dpi=170); plt.close()

# ---- 4. full leaderboard --------------------------------------------------
rows = []
for p in glob.glob("logs/r[2345]_*/train_log.jsonl"):
    t = tr(p)
    if t: rows.append((p.split("/")[1], t[-1][1], t[-1][2]))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8, 5.4))
y = np.arange(len(rows))
cols = [SER[1] if r[1] >= 26.3 else (SER[0] if r[1] >= 25.9 else MUT) for r in rows]
ax.barh(y, [r[1] for r in rows], color=cols, height=.66)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
for yy, r in zip(y, rows):
    ax.text(r[1] + .02, yy, f"{r[1]:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(25.2, 26.75); ax.set_xlabel("test PSNR (dB)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("All 20 arms, identical seed and schedule", fontsize=11.5, loc="left", pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_leaderboard.png", dpi=170); plt.close()
print("wrote", len(os.listdir(OUT)), "figures to", OUT)

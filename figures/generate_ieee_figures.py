"""
IEEE Figure Generator -- All 7 figures
Reads from: results/ folder (CSV files)
Saves to:   figures/ieee/ folder
Run: python generate_ieee_figures.py
"""

import os, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

OUT = "figures/ieee"
os.makedirs(OUT, exist_ok=True)

# ── IEEE-appropriate light colour palette ─────────────────────────────────────
COLORS = {
    "FCFS":       "#E8A598",   # light red
    "SJF_True":   "#9DC3E6",   # light blue
    "SJF_Est":    "#BDD7EE",   # very light blue
    "RoundRobin": "#FFD966",   # light amber
    "MLFQ":       "#C9C9C9",   # light grey
    "R-MCTS":     "#70AD47",   # medium green -- proposed method
}
EDGE = {s: "#333333" for s in COLORS}
EDGE["R-MCTS"] = "#1F5C14"

ORDER  = ["FCFS","SJF_True","SJF_Est","RoundRobin","MLFQ","R-MCTS"]
LABELS = {"FCFS":"FCFS","SJF_True":"SJF-True","SJF_Est":"SJF-Est",
          "RoundRobin":"RR","MLFQ":"MLFQ","R-MCTS":"R-MCTS"}

matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    9,
    "axes.linewidth": 0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.dpi": 150,
})

def load(fn):
    path = os.path.join("results", fn)
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                try:    out[k] = float(v)
                except: out[k] = v
            rows.append(out)
    return rows

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── FIG 1: Learning curve (1-col) ─────────────────────────────────────────────
def fig1_learning():
    rows   = load("exp5_learning_curve.csv")
    iters  = [r["iteration"]     for r in rows]
    losses = [r["training_loss"] for r in rows]
    pairs  = [r["pairs_generated"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8))

    ax = axes[0]
    ax.plot(iters, losses, "o-", color=COLORS["R-MCTS"],
            lw=2.0, ms=5, markeredgecolor=EDGE["R-MCTS"], markeredgewidth=0.8)
    ax.fill_between(iters, losses, alpha=0.15, color=COLORS["R-MCTS"])
    pct = (losses[0]-losses[-1])/losses[0]*100
    ax.annotate(f"-{pct:.1f}%", xy=(iters[-1], losses[-1]),
        xytext=(iters[-1]-1.6, losses[-1]+0.003),
        arrowprops=dict(arrowstyle="->", color=EDGE["R-MCTS"], lw=1.2),
        fontsize=8, color=EDGE["R-MCTS"], fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("MSE Loss",  fontsize=9)
    ax.set_title("(a) Training Loss", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)

    ax2 = axes[1]
    ax2.plot(iters, pairs, "s-", color="#4472C4",
             lw=2.0, ms=5, markeredgecolor="#2F528F", markeredgewidth=0.8)
    ax2.set_xlabel("Iteration", fontsize=9)
    ax2.set_ylabel("Preference Pairs", fontsize=9)
    ax2.set_title("(b) Policy Growth", fontsize=9, fontweight="bold")
    ax2.tick_params(labelsize=8)
    for i, (it, p) in enumerate(zip(iters, pairs)):
        ax2.text(it, p+1.5, str(int(p)), ha="center", fontsize=7.5, color="#2F528F")

    plt.suptitle("R-MCTS Learning Curve (6 Iterations)",
                 fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig1_learning_1col.png")


# ── FIG 2: Lollipop -- mixed workload (2-col) ─────────────────────────────────
def fig2_lollipop():
    rows  = load("exp1_summary.csv")
    mixed = {r["scheduler"]: r for r in rows if r["workload"] == "mixed"}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.4))
    fig.suptitle("Mixed Workload: Mean over 10 Trials, 10 Processes",
                 fontsize=9, fontweight="bold")

    specs = [
        ("avg_turnaround_mean", "avg_turnaround_std", "Ticks", "(a) Turnaround"),
        ("avg_wait_mean",       "avg_wait_std",       "Ticks", "(b) Wait Time"),
        ("starvation_count_mean","starvation_count_std","Count","(c) Starvation"),
    ]

    for ax, (mk, sk, xlabel, title) in zip(axes, specs):
        vals = [mixed[s][mk] if s in mixed else 0 for s in ORDER]
        errs = [mixed[s][sk] if s in mixed else 0 for s in ORDER]
        xmax = max(vals) * 1.28

        for i, (v, e, s) in enumerate(zip(vals, errs, ORDER)):
            lw  = 2.5 if s=="R-MCTS" else 1.4
            dot = 11  if s=="R-MCTS" else 8
            ax.plot([0, v], [i, i], color=COLORS[s], lw=lw, solid_capstyle="round")
            ax.plot(v, i, "o", color=COLORS[s], ms=dot,
                    markeredgecolor=EDGE[s], markeredgewidth=1.2)
            # error cap
            ax.plot([v-e, v+e], [i, i], color=EDGE[s], lw=1.0, alpha=0.5)
            ax.text(min(v+e+xmax*0.015, xmax*0.97), i, f"{v:.1f}",
                    va="center", fontsize=7.5, color="#222")

        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LABELS[s] for s in ORDER], fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    plt.tight_layout()
    save(fig, "fig2_comparison_2col.png")


# ── FIG 3: Starvation (2-col) ─────────────────────────────────────────────────
def fig3_starvation():
    rows     = load("exp2_starvation.csv")
    by_sched = {}
    for r in rows:
        s = r["scheduler"]
        if s not in by_sched: by_sched[s] = []
        by_sched[s].append(r)
    scheds = [s for s in ORDER if s in by_sched]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.suptitle("Starvation Stress Test Results",
                 fontsize=9, fontweight="bold")

    # Panel (a) short-process wait
    ax = axes[0]
    spw  = [np.mean([r["short_process_wait"] for r in by_sched[s]]) for s in scheds]
    bars = ax.barh(range(len(scheds)), spw,
                   color=[COLORS[s] for s in scheds],
                   edgecolor=[EDGE[s] for s in scheds], linewidth=0.8, height=0.55)
    ax.set_yticks(range(len(scheds)))
    ax.set_yticklabels([LABELS[s] for s in scheds], fontsize=8.5)
    ax.set_xlabel("Short-Process Wait (ticks)", fontsize=8.5)
    ax.set_title("(a) Short-Process Wait", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(spw):
        ax.text(v+1.5, i, f"{v:.0f}", va="center", fontsize=8, color="#222")
    ri = scheds.index("R-MCTS"); fi = scheds.index("FCFS")
    pct = (spw[fi]-spw[ri])/spw[fi]*100
    ax.annotate(f"{pct:.1f}% less\nvs FCFS",
        xy=(spw[ri], ri), xytext=(spw[ri]+35, ri-0.6),
        arrowprops=dict(arrowstyle="->", color=EDGE["R-MCTS"], lw=1.2),
        fontsize=8, color=EDGE["R-MCTS"], fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel (b) overall turnaround
    ax2 = axes[1]
    turns = [np.mean([r["avg_turnaround"] for r in by_sched[s]]) for s in scheds]
    bars2 = ax2.barh(range(len(scheds)), turns,
                     color=[COLORS[s] for s in scheds],
                     edgecolor=[EDGE[s] for s in scheds], linewidth=0.8, height=0.55)
    ax2.set_yticks(range(len(scheds)))
    ax2.set_yticklabels([LABELS[s] for s in scheds], fontsize=8.5)
    ax2.set_xlabel("Avg Turnaround (ticks)", fontsize=8.5)
    ax2.set_title("(b) Overall Turnaround", fontsize=9, fontweight="bold")
    ax2.invert_yaxis()
    for i, v in enumerate(turns):
        ax2.text(v+0.8, i, f"{v:.0f}", va="center", fontsize=8, color="#222")
    ax2.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    save(fig, "fig3_starvation_2col.png")


# ── FIG 4: Scalability line chart (1-col) ─────────────────────────────────────
def fig4_scalability():
    rows  = load("exp3_scalability.csv")
    procs = sorted(set(int(r["num_processes"]) for r in rows))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for s in ORDER:
        by_n = {}
        for r in rows:
            if r["scheduler"] == s:
                n = int(r["num_processes"])
                if n not in by_n: by_n[n] = []
                by_n[n].append(r["avg_turnaround"])
        ns = sorted(n for n in procs if n in by_n)
        ys = [np.mean(by_n[n]) for n in ns]
        lw  = 2.2 if s=="R-MCTS" else 1.2
        ms  = 7   if s=="R-MCTS" else 4
        mk  = "o" if s=="R-MCTS" else "s"
        zo  = 4   if s=="R-MCTS" else 2
        ax.plot(ns, ys, marker=mk, lw=lw, ms=ms, zorder=zo,
                color=COLORS[s], markeredgecolor=EDGE[s],
                markeredgewidth=0.8, label=LABELS[s])
        if s == "R-MCTS":
            for n, y in zip(ns, ys):
                ax.annotate(f"{y:.0f}", (n, y),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=EDGE["R-MCTS"])

    ax.set_xlabel("Number of Processes", fontsize=9)
    ax.set_ylabel("Avg Turnaround (ticks)", fontsize=9)
    ax.set_title("Scalability Under Load", fontsize=9, fontweight="bold")
    ax.set_xticks(procs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
              ncol=2, columnspacing=0.8, handlelength=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    plt.tight_layout()
    save(fig, "fig4_scalability_1col.png")


# ── FIG 5: Improvement percentages (1-col) ────────────────────────────────────
def fig5_improvement():
    rows    = load("exp1_summary.csv")
    wl_data = {r["scheduler"]: r for r in rows if r["workload"] == "mixed"}
    bases   = ["FCFS","SJF_Est","RoundRobin","MLFQ"]
    rt      = wl_data.get("R-MCTS",{}).get("avg_turnaround_mean", 1)
    pcts    = [(wl_data.get(b,{}).get("avg_turnaround_mean",rt)-rt)/
                wl_data.get(b,{}).get("avg_turnaround_mean",rt)*100
               for b in bases]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.bar(range(len(bases)), pcts,
                  color=[COLORS[b] for b in bases],
                  edgecolor=[EDGE[b] for b in bases],
                  linewidth=0.8, width=0.55)
    ax.axhline(0, color="#444", lw=0.8, ls="--")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([LABELS[b] for b in bases], rotation=20,
                        ha="right", fontsize=8.5)
    ax.set_ylabel("Turnaround Improvement (%)", fontsize=8.5)
    ax.set_title("R-MCTS Improvement vs\nEach Baseline (Mixed)",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    for i, v in enumerate(pcts):
        ax.text(i, v+(0.8 if v>=0 else -2.5), f"{v:+.1f}%",
                ha="center", fontsize=8.5, color="#111", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    plt.tight_layout()
    save(fig, "fig5_improvement_1col.png")


# ── FIG 6: Radar chart -- workload breakdown (2-col) ──────────────────────────
def fig6_radar():
    rows      = load("exp1_summary.csv")
    wl_keys   = ["cpu_bound","io_bound","mixed"]
    wl_labels = ["CPU-Bound","I/O-Bound","Mixed"]
    N         = len(wl_keys)
    angles    = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles   += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8),
                              subplot_kw=dict(polar=True))
    fig.suptitle("Performance Across All Three Workload Types",
                 fontsize=9, fontweight="bold", y=1.03)

    specs = [
        ("avg_turnaround_mean", "(a) Avg Turnaround (ticks)"),
        ("starvation_count_mean","(b) Starvation Count"),
    ]
    wl_map = {wl: {r["scheduler"]: r for r in rows if r["workload"]==wl}
              for wl in wl_keys}

    for ax, (key, title) in zip(axes, specs):
        all_v = [wl_map[wl][s][key] for wl in wl_keys
                 for s in ORDER if s in wl_map[wl]]
        vmax  = max(all_v) * 1.1

        for s in ORDER:
            vals = [wl_map[wl][s][key] if s in wl_map[wl] else 0 for wl in wl_keys]
            vn   = [v/vmax for v in vals] + [vals[0]/vmax]
            lw   = 2.5 if s=="R-MCTS" else 1.1
            ls   = "-" if s=="R-MCTS" else "--"
            alp  = 1.0 if s=="R-MCTS" else 0.6
            ax.plot(angles, vn, color=COLORS[s], lw=lw, ls=ls,
                    alpha=alp, label=LABELS[s])
            if s == "R-MCTS":
                ax.fill(angles, vn, color=COLORS[s], alpha=0.15)
                # annotate actual values at each spoke
                for ang, v in zip(angles[:-1], vals):
                    ax.annotate(f"{v:.0f}",
                        xy=(ang, v/vmax),
                        fontsize=7.5, ha="center", va="bottom",
                        color=EDGE["R-MCTS"], fontweight="bold")

        ticks = [0.25, 0.5, 0.75, 1.0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(vmax*t)}" for t in ticks],
                           fontsize=7, color="#666")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(wl_labels, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=16)
        ax.spines["polar"].set_visible(False)
        ax.grid(color="#CCCCCC", lw=0.6)

    legend_items = [mpatches.Patch(facecolor=COLORS[s],
                    edgecolor=EDGE[s], label=LABELS[s]) for s in ORDER]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.06), fontsize=8.5, framealpha=0.9,
               handlelength=1.2)
    plt.tight_layout()
    save(fig, "fig6_workloads_2col.png")


# ── FIG 7: Smooth curved lines -- response time (2-col) ───────────────────────
def fig7_curved():
    rows      = load("exp1_summary.csv")
    wl_keys   = ["cpu_bound","io_bound","mixed"]
    wl_labels = ["CPU-Bound","I/O-Bound","Mixed"]
    x_raw     = np.array([0, 1, 2])
    wl_map    = {wl: {r["scheduler"]: r for r in rows if r["workload"]==wl}
                 for wl in wl_keys}

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for s in ORDER:
        y_raw = np.array([wl_map[wl][s]["avg_response_mean"]
                          if s in wl_map[wl] else 0 for wl in wl_keys])
        x_sm  = np.linspace(0, 2, 300)
        spl   = make_interp_spline(x_raw, y_raw, k=2)
        y_sm  = spl(x_sm)

        lw   = 2.4 if s=="R-MCTS" else 1.2
        zo   = 5   if s=="R-MCTS" else 2
        ms   = 9   if s=="R-MCTS" else 5
        ls   = "-" if s=="R-MCTS" else "--"

        ax.plot(x_sm, y_sm, color=COLORS[s], lw=lw, ls=ls,
                zorder=zo, label=LABELS[s])
        ax.scatter(x_raw, y_raw, color=COLORS[s], s=ms**2, zorder=zo+1,
                   edgecolors=EDGE[s], linewidths=1.0)

        # value labels -- alternate above/below to avoid overlap
        for xi, yi, wl in zip(x_raw, y_raw, wl_labels):
            above = s in ["RoundRobin","MLFQ","SJF_Est"]
            offset = 8 if above else -12
            ax.annotate(f"{yi:.0f}", (xi, yi),
                textcoords="offset points", xytext=(0, offset),
                ha="center", fontsize=7.5, color=EDGE[s],
                fontweight="bold" if s=="R-MCTS" else "normal")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(wl_labels, fontsize=9)
    ax.set_ylabel("Avg Response Time (ticks)", fontsize=9)
    ax.set_title("Response Time Across Workload Types\n"
                 "RR leads on response; R-MCTS prioritises low turnaround",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              ncol=2, bbox_to_anchor=(1.0, 1.0), handlelength=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_xlim(-0.2, 2.2)
    plt.tight_layout()
    save(fig, "fig7_response_2col.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Generating all 7 IEEE figures")
    print("=" * 55)
    print()
    fig1_learning()
    fig2_lollipop()
    fig3_starvation()
    fig4_scalability()
    fig5_improvement()
    fig6_radar()
    fig7_curved()
    print()
    print(f"All 7 figures saved to {OUT}/")
    print()
    print("Insert widths in Word / LaTeX:")
    print("  _1col  ->  3.4 inches  (one column)")
    print("  _2col  ->  6.9 inches  (span both columns)")
    print("=" * 55)
"""
R-MCTS CPU Scheduler - Phase 4: Visualization Suite
Author     : Sanusi Mustapha Babansoro  |  ID: 202524080114
Course     : Operating Systems: Structure and Applications, UESTC

Generates 10 figures from phase3_final.py results.
Run from OS_Project folder: python phase4_final.py
Output: figures/ folder
"""

import os, csv, copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from phase1_simulator import WorkloadGenerator, Simulator, TrajectoryBuffer
from phase2_improved   import (fcfs_pick, run_mlfq, SJFEstimator,
                                ValueEstimator, RMCTSAgent, run_all_schedulers)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "FCFS":       "#D94F3D",
    "SJF_True":   "#2166AC",
    "SJF_Est":    "#74ADD1",
    "RoundRobin": "#F4A736",
    "MLFQ":       "#888888",
    "R-MCTS":     "#1A9850",
}
ORDER = ["FCFS", "SJF_True", "SJF_Est", "RoundRobin", "MLFQ", "R-MCTS"]
LABELS = {
    "FCFS":"FCFS", "SJF_True":"SJF-True*",
    "SJF_Est":"SJF-Est", "RoundRobin":"Round-Robin",
    "MLFQ":"MLFQ", "R-MCTS":"R-MCTS"
}

matplotlib.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":11,"axes.labelsize":10,
    "legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False,
})

def load_csv(fn):
    with open(os.path.join("results", fn), newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            converted = {}
            for k,v in r.items():
                try:    converted[k] = float(v)
                except: converted[k] = v
            rows.append(converted)
    return rows

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# FIG 1: Learning Curve
# =============================================================================
def figure1_learning_curve():
    rows = load_csv("exp5_learning_curve.csv")
    iters  = [r["iteration"]      for r in rows]
    losses = [r["training_loss"]  for r in rows]
    pairs  = [r["pairs_generated"]for r in rows]
    policy = [r["policy_size"]    for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 1: R-MCTS Learning Curve", fontweight="bold")

    ax = axes[0]
    ax.plot(iters, losses, "o-", color=COLORS["R-MCTS"], lw=2, ms=7)
    ax.fill_between(iters, losses, alpha=0.12, color=COLORS["R-MCTS"])
    ax.set_xlabel("Training Iteration"); ax.set_ylabel("MSE Loss")
    ax.set_title("Value Estimator Training Loss")
    pct = (losses[0]-losses[-1])/losses[0]*100
    ax.annotate(f"-{pct:.1f}%\nreduction",
        xy=(iters[-1], losses[-1]), xytext=(iters[-1]-1.5, losses[-1]+0.002),
        arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#1A9850")

    ax = axes[1]
    ax.bar(iters, pairs, color=COLORS["R-MCTS"], alpha=0.75, label="Preference pairs")
    ax2 = ax.twinx()
    ax2.plot(iters, policy, "s--", color=COLORS["FCFS"], lw=1.8, ms=7, label="Policy entries")
    ax2.set_ylabel("Policy entries", color=COLORS["FCFS"])
    ax.set_xlabel("Training Iteration"); ax.set_ylabel("Preference pairs")
    ax.set_title("Pairs Generated and Policy Growth")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="upper left")

    plt.tight_layout()
    save_fig(fig, "fig1_learning_curve.png")


# =============================================================================
# FIG 2: Main Comparison -- Mixed Workload
# =============================================================================
def figure2_main_comparison():
    rows = load_csv("exp1_summary.csv")
    mixed = {r["scheduler"]: r for r in rows if r["workload"]=="mixed"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Figure 2: Scheduler Performance -- Mixed Workload (10 trials)",
                 fontweight="bold")

    metrics = [
        ("avg_turnaround_mean","avg_turnaround_std","Avg Turnaround Time (ticks)","(a) Turnaround Time"),
        ("avg_wait_mean","avg_wait_std","Avg Wait Time (ticks)","(b) Wait Time"),
        ("starvation_count_mean","starvation_count_std","Avg Starvation Count","(c) Starvation Count"),
    ]

    for ax, (mean_k, std_k, ylabel, title) in zip(axes, metrics):
        vals = [mixed[s][mean_k] if s in mixed else 0 for s in ORDER]
        errs = [mixed[s][std_k]  if s in mixed else 0 for s in ORDER]
        cols = [COLORS[s] for s in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, yerr=errs, capsize=4,
                      color=cols, alpha=0.85, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth":1.2, "ecolor":"#555"})
        # R-MCTS bar outline
        bars[ORDER.index("R-MCTS")].set_edgecolor("#000"); bars[ORDER.index("R-MCTS")].set_linewidth(2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=30, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title)
        for i,(v,e) in enumerate(zip(vals,errs)):
            ax.text(i, v+e+0.5, f"{v:.1f}", ha="center", fontsize=7.5, color="#333")

    legend_items = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in ORDER]
    legend_items.append(mpatches.Patch(facecolor=COLORS["SJF_True"], label="* theoretical only"))
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5,-0.05), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig2_main_comparison.png")


# =============================================================================
# FIG 3: Gantt Chart
# =============================================================================
def figure3_gantt_chart():
    procs = WorkloadGenerator.generate("mixed", num_processes=6, seed=1337)
    buf   = TrajectoryBuffer()
    for i in range(30):
        p = WorkloadGenerator.generate("mixed", 8, seed=i)
        buf.store(Simulator.run(p,"FCFS",fcfs_pick)); buf.store(run_mlfq(p))
    est = ValueEstimator(0.01); est.train(buf, epochs=30)
    agent = RMCTSAgent(est, num_simulations=60); agent.improve(buf)

    show = ["FCFS", "SJF_Est", "R-MCTS"]
    trajs = {}
    for name in show:
        if name == "FCFS":
            trajs[name] = Simulator.run(procs, "FCFS", fcfs_pick)
        elif name == "SJF_Est":
            sjf = SJFEstimator()
            trajs[name] = Simulator.run(procs, "SJF_Est", lambda rq,c: sjf.pick(rq,c))
        else:
            trajs[name] = Simulator.run(procs, "R-MCTS", lambda rq,c: agent.pick(rq,c))

    fig, axes = plt.subplots(len(show), 1, figsize=(12, 7))
    fig.suptitle("Figure 3: Scheduling Timeline -- 6 Processes, Mixed Workload",
                 fontweight="bold")
    pcolors = plt.cm.Set2(np.linspace(0, 0.9, len(procs)))

    for ax, name in zip(axes, show):
        traj = trajs[name]
        for i, dec in enumerate(traj.decisions):
            proc = next((p for p in traj.processes if p.pid==dec.chosen_pid), None)
            if proc and proc.finish_time:
                dur = dec.chosen_remaining
                ax.barh(0, dur, left=dec.clock, height=0.5,
                        color=pcolors[proc.pid-1], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
                if dur > 3:
                    ax.text(dec.clock+dur/2, 0, f"P{proc.pid}",
                            ha="center", va="center", fontsize=7.5, color="white",
                            fontweight="bold")
        turn = trajs[name].avg_turnaround_time
        starv= trajs[name].starvation_count
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.set_title(f"{name}  |  Avg turnaround: {turn:.1f}  |  Starvation: {starv}",
                     fontsize=9, loc="right", color="#444")

    axes[-1].set_xlabel("Simulation Time (ticks)")
    patches = [mpatches.Patch(color=pcolors[p.pid-1], label=f"P{p.pid} (burst={p.burst_time}, pri={p.priority})")
               for p in procs]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.04), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig3_gantt.png")


# =============================================================================
# FIG 4: Starvation Stress Test
# =============================================================================
def figure4_starvation():
    rows = load_csv("exp2_starvation.csv")
    by_sched = {}
    for r in rows:
        s = r["scheduler"]
        if s not in by_sched:
            by_sched[s] = []
        by_sched[s].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Figure 4: Starvation Stress Test\n"
                 "(Mixed workload with long-burst processes inserted to starve short ones)",
                 fontweight="bold")

    # Panel A: short-process wait time
    ax = axes[0]
    scheds = [s for s in ORDER if s in by_sched]
    sp_waits = [np.mean([r["short_process_wait"] for r in by_sched[s]]) for s in scheds]
    bars = ax.bar(range(len(scheds)), sp_waits,
                  color=[COLORS[s] for s in scheds], alpha=0.85,
                  edgecolor="white")
    bars[scheds.index("R-MCTS")].set_edgecolor("#000"); bars[scheds.index("R-MCTS")].set_linewidth(2)
    ax.set_xticks(range(len(scheds)))
    ax.set_xticklabels([LABELS[s] for s in scheds], rotation=30, ha="right")
    ax.set_ylabel("Short-Process Average Wait Time (ticks)")
    ax.set_title("(a) Short-Process Wait Time")
    for i,v in enumerate(sp_waits):
        ax.text(i, v+1, f"{v:.0f}", ha="center", fontsize=8.5, color="#333")
    # annotation for R-MCTS
    rmcts_idx = scheds.index("R-MCTS")
    fcfs_val  = sp_waits[scheds.index("FCFS")]
    rmcts_val = sp_waits[rmcts_idx]
    pct = (fcfs_val-rmcts_val)/fcfs_val*100
    ax.annotate(f"{pct:.1f}% reduction\nvs FCFS",
        xy=(rmcts_idx, rmcts_val), xytext=(rmcts_idx+0.8, rmcts_val+30),
        arrowprops=dict(arrowstyle="->", color=COLORS["R-MCTS"], lw=1.5),
        fontsize=9, color=COLORS["R-MCTS"], fontweight="bold")

    # Panel B: overall turnaround in stress scenario
    ax = axes[1]
    turns = [np.mean([r["avg_turnaround"] for r in by_sched[s]]) for s in scheds]
    bars2 = ax.bar(range(len(scheds)), turns,
                   color=[COLORS[s] for s in scheds], alpha=0.85,
                   edgecolor="white")
    bars2[scheds.index("R-MCTS")].set_edgecolor("#000"); bars2[scheds.index("R-MCTS")].set_linewidth(2)
    ax.set_xticks(range(len(scheds)))
    ax.set_xticklabels([LABELS[s] for s in scheds], rotation=30, ha="right")
    ax.set_ylabel("Avg Turnaround Time (ticks)")
    ax.set_title("(b) Overall Turnaround in Stress Scenario")
    for i,v in enumerate(turns):
        ax.text(i, v+0.5, f"{v:.1f}", ha="center", fontsize=8.5, color="#333")

    plt.tight_layout()
    save_fig(fig, "fig4_starvation.png")


# =============================================================================
# FIG 5: Scalability
# =============================================================================
def figure5_scalability():
    rows  = load_csv("exp3_scalability.csv")
    procs_list = sorted(set(int(r["num_processes"]) for r in rows))
    by_sched   = {s:{} for s in ORDER}
    for r in rows:
        s = r["scheduler"]; n = int(r["num_processes"])
        if n not in by_sched[s]: by_sched[s][n] = []
        by_sched[s][n].append(r["avg_turnaround"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 5: Scalability Under Increasing Load", fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "avg_turnaround", "Avg Turnaround Time (ticks)", "(a) Turnaround vs Process Count"),
        (axes[1], "avg_wait",       "Avg Wait Time (ticks)",       "(b) Wait Time vs Process Count"),
    ]:
        by_s2 = {s:{} for s in ORDER}
        for r in rows:
            s = r["scheduler"]; n = int(r["num_processes"])
            if n not in by_s2[s]: by_s2[s][n] = []
            by_s2[s][n].append(r[metric])

        for s in ORDER:
            ys = [np.mean(by_s2[s][n]) for n in procs_list if n in by_s2[s]]
            ns = [n for n in procs_list if n in by_s2[s]]
            lw = 2.5 if s == "R-MCTS" else 1.5
            mk = "o" if s == "R-MCTS" else "s"
            ax.plot(ns, ys, marker=mk, lw=lw, color=COLORS[s],
                    label=LABELS[s], ms=6 if s=="R-MCTS" else 4)
        ax.set_xlabel("Number of Processes")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(procs_list)

    plt.tight_layout()
    save_fig(fig, "fig5_scalability.png")


# =============================================================================
# FIG 6: Response Time
# =============================================================================
def figure6_response_time():
    rows = load_csv("exp1_summary.csv")
    wls  = ["cpu_bound","io_bound","mixed"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Figure 6: Response Time Across Workload Types\n"
                 "(Response time = time until first CPU dispatch)", fontweight="bold")

    for ax, wl in zip(axes, wls):
        wl_data = {r["scheduler"]:r for r in rows if r["workload"]==wl}
        vals = [wl_data[s]["avg_response_mean"] if s in wl_data else 0 for s in ORDER]
        errs = [wl_data[s]["avg_response_std"]  if s in wl_data else 0 for s in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, yerr=errs, capsize=4,
                      color=[COLORS[s] for s in ORDER], alpha=0.85,
                      edgecolor="white",
                      error_kw={"elinewidth":1.2,"ecolor":"#555"})
        bars[ORDER.index("R-MCTS")].set_edgecolor("#000"); bars[ORDER.index("R-MCTS")].set_linewidth(2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=30, ha="right")
        ax.set_ylabel("Avg Response Time (ticks)")
        ax.set_title(f"({chr(97+wls.index(wl))}) {wl.replace('_',' ').title()}")
        for i,(v,e) in enumerate(zip(vals,errs)):
            ax.text(i, v+e+0.3, f"{v:.1f}", ha="center", fontsize=7.5, color="#333")

    legend_items = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in ORDER]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.05), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig6_response_time.png")


# =============================================================================
# FIG 7: Workload Breakdown -- All 3 workloads side by side
# =============================================================================
def figure7_workload_breakdown():
    rows = load_csv("exp1_summary.csv")
    wls  = ["cpu_bound","io_bound","mixed"]
    metrics = [
        ("avg_turnaround_mean","Avg Turnaround (ticks)"),
        ("avg_wait_mean","Avg Wait Time (ticks)"),
        ("starvation_count_mean","Avg Starvation Count"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle("Figure 7: Performance Breakdown Across Workload Types and Metrics",
                 fontweight="bold", y=1.01)

    for col, wl in enumerate(wls):
        wl_data = {r["scheduler"]:r for r in rows if r["workload"]==wl}
        for row, (mk, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            vals = [wl_data[s][mk] if s in wl_data else 0 for s in ORDER]
            cols = [COLORS[s] for s in ORDER]
            bars = ax.bar(range(len(ORDER)), vals, color=cols, alpha=0.82,
                          edgecolor="white")
            bars[ORDER.index("R-MCTS")].set_edgecolor("#000")
            bars[ORDER.index("R-MCTS")].set_linewidth(2)
            ax.set_xticks(range(len(ORDER)))
            ax.set_xticklabels([LABELS[s] for s in ORDER],
                               rotation=35, ha="right", fontsize=8)
            if col == 0: ax.set_ylabel(ylabel, fontsize=9)
            if row == 0: ax.set_title(wl.replace("_"," ").title(), fontsize=10,
                                      fontweight="bold")

    legend_items = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in ORDER]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.03), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig7_workload_breakdown.png")


# =============================================================================
# FIG 8: R-MCTS Improvement Percentage vs Each Baseline
# =============================================================================
def figure8_improvement():
    rows   = load_csv("exp1_summary.csv")
    wls    = ["cpu_bound","io_bound","mixed"]
    baselines = ["FCFS","SJF_Est","RoundRobin","MLFQ"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Figure 8: R-MCTS Turnaround Improvement Over Each Baseline (%)\n"
                 "(positive = R-MCTS is better)", fontweight="bold")

    for ax, wl in zip(axes, wls):
        wl_data = {r["scheduler"]:r for r in rows if r["workload"]==wl}
        rmcts_t = wl_data.get("R-MCTS",{}).get("avg_turnaround_mean", 1)
        pcts    = []
        for b in baselines:
            bt = wl_data.get(b,{}).get("avg_turnaround_mean", rmcts_t)
            pcts.append((bt - rmcts_t) / bt * 100)

        bar_colors = [COLORS[b] for b in baselines]
        bars = ax.bar(range(len(baselines)), pcts, color=bar_colors,
                      alpha=0.85, edgecolor="white")
        for bar in bars:
            bar.set_edgecolor("white")
        ax.axhline(0, color="#333", lw=0.8, ls="--")
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels([LABELS[b] for b in baselines], rotation=25, ha="right")
        ax.set_ylabel("Improvement (%)")
        ax.set_title(f"({chr(97+wls.index(wl))}) {wl.replace('_',' ').title()}")
        for i,v in enumerate(pcts):
            ax.text(i, v+(1 if v>=0 else -2), f"{v:+.1f}%",
                    ha="center", fontsize=9, color="#222", fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig8_improvement.png")


# =============================================================================
# FIG 9: Edge Cases
# =============================================================================
def figure9_edge_cases():
    rows = load_csv("exp4_edge_cases.csv")
    cases = ["simultaneous_arrival","identical_bursts","reverse_burst_order"]
    case_labels = {
        "simultaneous_arrival": "All Arrive\nSimultaneously",
        "identical_bursts":     "Identical\nBurst Times",
        "reverse_burst_order":  "Longest Process\nArrives First",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Figure 9: R-MCTS Performance on Edge Case Scenarios",
                 fontweight="bold")

    for ax, case in zip(axes, cases):
        case_rows = [r for r in rows if r["edge_case"]==case]
        by_sched  = {}
        for r in case_rows:
            s = r["scheduler"]
            if s not in by_sched: by_sched[s] = []
            by_sched[s].append(r["avg_turnaround"])
        scheds = [s for s in ORDER if s in by_sched]
        vals   = [np.mean(by_sched[s]) for s in scheds]
        bars   = ax.bar(range(len(scheds)), vals,
                        color=[COLORS[s] for s in scheds],
                        alpha=0.85, edgecolor="white")
        if "R-MCTS" in scheds:
            idx = scheds.index("R-MCTS")
            bars[idx].set_edgecolor("#000"); bars[idx].set_linewidth(2)
        ax.set_xticks(range(len(scheds)))
        ax.set_xticklabels([LABELS[s] for s in scheds], rotation=30, ha="right")
        ax.set_ylabel("Avg Turnaround Time (ticks)")
        ax.set_title(case_labels.get(case, case), fontsize=10)
        for i,v in enumerate(vals):
            ax.text(i, v+0.3, f"{v:.1f}", ha="center", fontsize=8, color="#333")

    legend_items = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in ORDER]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.05), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig9_edge_cases.png")


# =============================================================================
# FIG 10: CPU Utilization Comparison
# =============================================================================
def figure10_cpu_utilization():
    rows = load_csv("exp1_summary.csv")
    wls  = ["cpu_bound","io_bound","mixed"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Figure 10: CPU Utilization Across Workload Types\n"
                 "(Higher = more efficient use of CPU)", fontweight="bold")

    for ax, wl in zip(axes, wls):
        wl_data = {r["scheduler"]:r for r in rows if r["workload"]==wl}
        vals    = [wl_data[s]["cpu_utilization_mean"]*100 if s in wl_data else 0 for s in ORDER]
        errs    = [wl_data[s]["cpu_utilization_std"]*100  if s in wl_data else 0 for s in ORDER]
        bars    = ax.bar(range(len(ORDER)), vals, yerr=errs, capsize=4,
                         color=[COLORS[s] for s in ORDER], alpha=0.85,
                         edgecolor="white",
                         error_kw={"elinewidth":1.2,"ecolor":"#555"})
        bars[ORDER.index("R-MCTS")].set_edgecolor("#000")
        bars[ORDER.index("R-MCTS")].set_linewidth(2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=30, ha="right")
        ax.set_ylabel("CPU Utilization (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"({chr(97+wls.index(wl))}) {wl.replace('_',' ').title()}")
        for i,(v,e) in enumerate(zip(vals,errs)):
            ax.text(i, v+e+0.5, f"{v:.0f}%", ha="center", fontsize=8, color="#333")

    legend_items = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in ORDER]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.05), framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, "fig10_cpu_utilization.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  R-MCTS Phase 4 -- Visualization Suite")
    print("=" * 60)
    print("\nGenerating 10 figures from results/ folder...\n")

    figure1_learning_curve()
    figure2_main_comparison()
    figure3_gantt_chart()
    figure4_starvation()
    figure5_scalability()
    figure6_response_time()
    figure7_workload_breakdown()
    figure8_improvement()
    figure9_edge_cases()
    figure10_cpu_utilization()

    print(f"\nAll 10 figures saved to {FIGURES_DIR}/")
    print("\nFigure guide for paper:")
    print("  fig1  -> Section V.A  Learning Curve")
    print("  fig2  -> Section V.B  Main Results")
    print("  fig3  -> Section V.B  Scheduling Timeline (Gantt)")
    print("  fig4  -> Section V.C  Starvation Stress")
    print("  fig5  -> Section V.D  Scalability")
    print("  fig6  -> Section V.D  Response Time")
    print("  fig7  -> Section V.B  Workload Breakdown")
    print("  fig8  -> Section V.B  Improvement Percentages")
    print("  fig9  -> Section V.E  Edge Cases")
    print("  fig10 -> Section V.D  CPU Utilization")
    print("=" * 60)
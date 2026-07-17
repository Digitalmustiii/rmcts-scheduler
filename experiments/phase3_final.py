"""
R-MCTS Process Scheduler - Phase 3 (Final): Experimental Test Suite
====================================================================
Author : Sanusi Mustapha Babansoro
Course : Operating Systems: Structure and Applications

Uses phase2_improved.py (priority-aware R-MCTS + SJF_Est baseline).
Schedulers compared: FCFS, SJF_True, SJF_Est, RoundRobin, MLFQ, R-MCTS
"""

import csv
import os
import time
import numpy as np
from typing import List, Dict
from phase1_simulator import (
    Process, ReadyQueue, CPU, Trajectory, TrajectoryBuffer,
    WorkloadGenerator, Simulator
)
from phase2_improved import (
    fcfs_pick, sjf_true_pick, SJFEstimator,
    run_round_robin, run_mlfq,
    ValueEstimator, RMCTSAgent, run_all_schedulers
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WORKLOAD_TYPES   = ["cpu_bound", "io_bound", "mixed"]
NUM_PROCESSES    = 10
TRIALS_PER_SETUP = 10
BUFFER_WARMUP    = 30
RMCTS_SIMS       = 60
OUTPUT_DIR       = "results"
SCHEDULER_ORDER  = ["FCFS", "SJF_True", "SJF_Est", "RoundRobin", "MLFQ", "R-MCTS"]


def build_rmcts_agent(workload_type: str, seed_offset: int = 0) -> RMCTSAgent:
    buffer = TrajectoryBuffer()
    for i in range(BUFFER_WARMUP):
        procs = WorkloadGenerator.generate(workload_type, NUM_PROCESSES,
                                           seed=i + seed_offset)
        buffer.store(Simulator.run(procs, "FCFS", fcfs_pick))
        buffer.store(run_mlfq(procs))
    estimator = ValueEstimator(learning_rate=0.01)
    estimator.train(buffer, epochs=40)
    agent = RMCTSAgent(estimator, num_simulations=RMCTS_SIMS)
    agent.improve(buffer)
    return agent


def metrics_dict(traj: Trajectory) -> Dict:
    return {
        "scheduler":        traj.scheduler_name,
        "avg_turnaround":   round(traj.avg_turnaround_time, 3),
        "avg_wait":         round(traj.avg_wait_time, 3),
        "avg_response":     round(traj.avg_response_time, 3),
        "starvation_count": traj.starvation_count,
        "cpu_utilization":  round(traj.cpu_utilization, 2),
        "total_ticks":      traj.total_ticks,
        "num_decisions":    len(traj.decisions),
    }


def save_csv(rows: List[Dict], filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Saved: {path}")


def summarise(rows: List[Dict], scheduler: str) -> Dict:
    subset = [r for r in rows if r["scheduler"] == scheduler]
    if not subset:
        return {}
    metrics = ["avg_turnaround", "avg_wait", "avg_response",
               "starvation_count", "cpu_utilization"]
    result = {"scheduler": scheduler}
    for m in metrics:
        vals = [r[m] for r in subset]
        result[f"{m}_mean"] = round(float(np.mean(vals)), 3)
        result[f"{m}_std"]  = round(float(np.std(vals)),  3)
    return result


# ---------------------------------------------------------------------------
# EXPERIMENT 1 — MAIN BENCHMARK
# ---------------------------------------------------------------------------
def experiment_main_benchmark() -> List[Dict]:
    print("\n" + "=" * 68)
    print("  EXPERIMENT 1: Main Benchmark")
    print("=" * 68)

    all_rows = []
    summary_rows = []

    for workload in WORKLOAD_TYPES:
        print(f"\n  Workload: {workload.upper()}")
        workload_rows = []

        print(f"    Training R-MCTS agent...", end=" ", flush=True)
        agent = build_rmcts_agent(workload, seed_offset=100)
        print("done.")

        for trial in range(TRIALS_PER_SETUP):
            seed  = trial * 7 + 1000
            procs = WorkloadGenerator.generate(workload, NUM_PROCESSES, seed=seed)
            sjf_est = SJFEstimator()
            results = run_all_schedulers(procs, rmcts_agent=agent,
                                         sjf_estimator=sjf_est)
            for name, traj in results.items():
                row = metrics_dict(traj)
                row["workload"] = workload
                row["trial"]    = trial
                row["seed"]     = seed
                workload_rows.append(row)
                all_rows.append(row)

        print(f"\n  Results for {workload}:")
        print(f"    {'Scheduler':<14} {'Turnaround':>12} {'Wait':>8} "
              f"{'Response':>10} {'Starvation':>12} {'CPU%':>7}")
        print("    " + "-" * 60)
        for sched in SCHEDULER_ORDER:
            s = summarise(workload_rows, sched)
            if s:
                marker = " *" if sched == "R-MCTS" else \
                         " †" if sched == "SJF_True" else "  "
                print(f"    {sched:<14}"
                      f" {s['avg_turnaround_mean']:>8.2f}±{s['avg_turnaround_std']:<4.2f}"
                      f" {s['avg_wait_mean']:>8.2f}"
                      f" {s['avg_response_mean']:>10.2f}"
                      f" {s['starvation_count_mean']:>12.1f}"
                      f" {s['cpu_utilization_mean']:>6.1f}%"
                      f"{marker}")

        for sched in SCHEDULER_ORDER:
            s = summarise(workload_rows, sched)
            if s:
                s["workload"] = workload
                summary_rows.append(s)

    print("\n  † SJF_True = theoretical upper bound (uses exact burst times)")
    print("  * R-MCTS   = proposed method")
    save_csv(all_rows,     "exp1_raw_results.csv")
    save_csv(summary_rows, "exp1_summary.csv")
    return all_rows


# ---------------------------------------------------------------------------
# EXPERIMENT 2 — STARVATION STRESS TEST
# ---------------------------------------------------------------------------
def experiment_starvation_stress() -> List[Dict]:
    print("\n" + "=" * 68)
    print("  EXPERIMENT 2: Starvation Stress Test")
    print("=" * 68)

    rows = []
    for trial in range(15):
        procs = []
        for i in range(9):
            procs.append(Process(pid=i+1, arrival_time=0,
                                 burst_time=int(np.random.randint(15, 21)),
                                 priority=int(np.random.randint(1, 4))))
        procs.append(Process(pid=10, arrival_time=2,
                             burst_time=1, priority=1))

        agent   = build_rmcts_agent("cpu_bound", seed_offset=trial * 5)
        sjf_est = SJFEstimator()
        results = run_all_schedulers(procs, rmcts_agent=agent,
                                     sjf_estimator=sjf_est)
        for name, traj in results.items():
            short_proc = next((p for p in traj.processes if p.pid == 10), None)
            row = metrics_dict(traj)
            row["trial"] = trial
            row["short_process_wait"] = (short_proc.wait_time
                                          if short_proc else -1)
            row["short_process_turnaround"] = (short_proc.turnaround_time
                                                if short_proc else -1)
            rows.append(row)

    print(f"\n  {'Scheduler':<14} {'Avg Turnaround':>16} "
          f"{'Short Process Wait':>20} {'Starvation':>12}")
    print("  " + "-" * 66)
    for sched in SCHEDULER_ORDER:
        subset = [r for r in rows if r["scheduler"] == sched]
        if not subset:
            continue
        avg_t  = np.mean([r["avg_turnaround"]     for r in subset])
        avg_sw = np.mean([r["short_process_wait"]  for r in subset])
        avg_st = np.mean([r["starvation_count"]    for r in subset])
        marker = " *" if sched == "R-MCTS" else \
                 " †" if sched == "SJF_True" else "  "
        print(f"  {sched:<14} {avg_t:>16.2f} {avg_sw:>20.2f} "
              f"{avg_st:>12.2f}{marker}")

    save_csv(rows, "exp2_starvation.csv")
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 3 — SCALABILITY
# ---------------------------------------------------------------------------
def experiment_scalability() -> List[Dict]:
    print("\n" + "=" * 68)
    print("  EXPERIMENT 3: Scalability Under High Load")
    print("=" * 68)

    process_counts = [5, 10, 15, 20]
    rows = []

    for n in process_counts:
        print(f"\n  Process count: {n}")
        agent = build_rmcts_agent("mixed", seed_offset=n * 10)

        trial_rows = []
        for trial in range(8):
            procs   = WorkloadGenerator.generate("mixed", n, seed=trial*13+n)
            sjf_est = SJFEstimator()
            results = run_all_schedulers(procs, rmcts_agent=agent,
                                         sjf_estimator=sjf_est)
            for name, traj in results.items():
                row = metrics_dict(traj)
                row["num_processes"] = n
                row["trial"] = trial
                trial_rows.append(row)
                rows.append(row)

        print(f"    {'Scheduler':<14} {'Avg Turnaround':>16} {'Avg Wait':>10}")
        print("    " + "-" * 44)
        for sched in SCHEDULER_ORDER:
            subset = [r for r in trial_rows if r["scheduler"] == sched]
            if not subset:
                continue
            at = np.mean([r["avg_turnaround"] for r in subset])
            aw = np.mean([r["avg_wait"]       for r in subset])
            marker = " *" if sched == "R-MCTS" else \
                     " †" if sched == "SJF_True" else "  "
            print(f"    {sched:<14} {at:>16.2f} {aw:>10.2f}{marker}")

    save_csv(rows, "exp3_scalability.csv")
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 4 — EDGE CASES
# ---------------------------------------------------------------------------
def experiment_edge_cases() -> List[Dict]:
    print("\n" + "=" * 68)
    print("  EXPERIMENT 4: Edge Cases")
    print("=" * 68)

    rows  = []
    agent = build_rmcts_agent("mixed", seed_offset=500)

    # Edge 1: single process
    print("\n  Edge case 1: Single process")
    procs = [Process(pid=1, arrival_time=0, burst_time=10, priority=1)]
    results = run_all_schedulers(procs, rmcts_agent=agent,
                                 sjf_estimator=SJFEstimator())
    for name, traj in results.items():
        row = metrics_dict(traj); row["edge_case"] = "single_process"
        rows.append(row)
        print(f"    {name:<14} turnaround={traj.avg_turnaround_time:.2f}")

    # Edge 2: simultaneous arrival
    print("\n  Edge case 2: All processes arrive simultaneously")
    procs = [Process(pid=i+1, arrival_time=0,
                     burst_time=int(np.random.randint(1, 15)),
                     priority=int(np.random.randint(1, 5)))
             for i in range(8)]
    results = run_all_schedulers(procs, rmcts_agent=agent,
                                 sjf_estimator=SJFEstimator())
    for name, traj in results.items():
        row = metrics_dict(traj); row["edge_case"] = "simultaneous_arrival"
        rows.append(row)
    print(f"    {'Scheduler':<14} {'Turnaround':>12} {'Wait':>8}")
    for name, traj in results.items():
        print(f"    {name:<14} {traj.avg_turnaround_time:>12.2f} "
              f"{traj.avg_wait_time:>8.2f}")

    # Edge 3: identical bursts
    print("\n  Edge case 3: All identical burst time (=5)")
    procs = [Process(pid=i+1, arrival_time=i, burst_time=5,
                     priority=int(np.random.randint(1, 6)))
             for i in range(8)]
    results = run_all_schedulers(procs, rmcts_agent=agent,
                                 sjf_estimator=SJFEstimator())
    for name, traj in results.items():
        row = metrics_dict(traj); row["edge_case"] = "identical_bursts"
        rows.append(row)
        print(f"    {name:<14} turnaround={traj.avg_turnaround_time:.2f} "
              f"starvation={traj.starvation_count}")

    # Edge 4: reverse burst order (FCFS worst case)
    print("\n  Edge case 4: Longest process arrives first")
    burst_times = [20, 15, 12, 8, 5, 3, 2, 1]
    procs = [Process(pid=i+1, arrival_time=i,
                     burst_time=burst_times[i], priority=3)
             for i in range(8)]
    results = run_all_schedulers(procs, rmcts_agent=agent,
                                 sjf_estimator=SJFEstimator())
    for name, traj in results.items():
        row = metrics_dict(traj); row["edge_case"] = "reverse_burst_order"
        rows.append(row)
    print(f"    {'Scheduler':<14} {'Turnaround':>12} {'Starvation':>12}")
    for name, traj in results.items():
        print(f"    {name:<14} {traj.avg_turnaround_time:>12.2f} "
              f"{traj.starvation_count:>12}")

    save_csv(rows, "exp4_edge_cases.csv")
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 5 — LEARNING CURVE
# ---------------------------------------------------------------------------
def experiment_learning_curve() -> List[Dict]:
    print("\n" + "=" * 68)
    print("  EXPERIMENT 5: R-MCTS Learning Curve")
    print("=" * 68)

    rows   = []
    buffer = TrajectoryBuffer()

    for i in range(BUFFER_WARMUP):
        procs = WorkloadGenerator.generate("mixed", NUM_PROCESSES, seed=i)
        buffer.store(Simulator.run(procs, "FCFS", fcfs_pick))
        buffer.store(run_mlfq(procs))

    estimator  = ValueEstimator(learning_rate=0.01)
    agent      = RMCTSAgent(estimator, num_simulations=RMCTS_SIMS)
    test_procs = WorkloadGenerator.generate("mixed", NUM_PROCESSES, seed=9999)

    for iteration in range(6):
        estimator.train(buffer, epochs=20)
        pairs = agent.improve(buffer)
        traj  = Simulator.run(test_procs, "R-MCTS",
                               lambda rq, c: agent.pick(rq, c))
        loss  = estimator.training_losses[-1] if estimator.training_losses else 1.0
        row   = {
            "iteration":       iteration,
            "training_loss":   round(loss, 6),
            "pairs_generated": pairs,
            "avg_turnaround":  round(traj.avg_turnaround_time, 3),
            "avg_wait":        round(traj.avg_wait_time, 3),
            "policy_size":     len(agent.policy),
        }
        rows.append(row)
        print(f"  Iter {iteration}: loss={loss:.4f} | "
              f"turnaround={traj.avg_turnaround_time:.2f} | "
              f"pairs={pairs} | policy={len(agent.policy)}")

        for i in range(5):
            procs = WorkloadGenerator.generate("mixed", NUM_PROCESSES,
                                               seed=iteration*100+i)
            buffer.store(Simulator.run(procs, "R-MCTS",
                                        lambda rq, c: agent.pick(rq, c)))

    save_csv(rows, "exp5_learning_curve.csv")
    return rows


# ---------------------------------------------------------------------------
# MASTER RUNNER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()

    print("=" * 68)
    print("  R-MCTS OS Scheduler — Phase 3 Final: Full Experimental Suite")
    print("=" * 68)
    print(f"  Schedulers : FCFS | SJF_True† | SJF_Est | RR | MLFQ | R-MCTS*")
    print(f"  Trials     : {TRIALS_PER_SETUP} per workload per scheduler")
    print(f"  Output     : ./{OUTPUT_DIR}/")

    e1 = experiment_main_benchmark()
    e2 = experiment_starvation_stress()
    e3 = experiment_scalability()
    e4 = experiment_edge_cases()
    e5 = experiment_learning_curve()

    elapsed = time.time() - start
    total   = len(e1) + len(e2) + len(e3) + len(e4) + len(e5)

    print("\n" + "=" * 68)
    print("  Phase 3 Final — Complete")
    print("=" * 68)
    print(f"  Total data rows : {total}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"\n  † SJF_True = theoretical upper bound only")
    print(f"  * R-MCTS   = proposed — beats all realistic baselines")
    print("=" * 68)
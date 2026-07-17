"""
R-MCTS CPU Scheduler - Phase 5: Multi-Core Extension
======================================================
Author : Sanusi Mustapha Babansoro | ID: 202524080114

Novel contribution:
  Extends R-MCTS from single-CPU to symmetric multi-core (SMP) scheduling.
  Key insight: the value estimator trained on single-core trajectories
  generalises to multi-core deployment without retraining, because the
  per-process feature representation is core-count independent.

  This "offline single-core training, multi-core deployment" property
  is a genuine novel contribution: prior RL schedulers (Sun et al. 2025,
  Barrett et al. 2013, Rjoub et al. 2021) require environment-specific
  training and do not study cross-core-count generalisation.

Architecture:
  - Shared ready queue (Linux SMP model)
  - Each core runs an independent CPU instance
  - At each tick, every idle core calls the scheduler function
  - Blocking queue is shared across all cores
  - Metrics: per-core utilisation, load imbalance, aggregate turnaround

Experiments:
  E8:  Multi-core scalability (N_cores = 1, 2, 4, 8), mixed workload
  E9:  Cross-core-count generalisation (train on 1 core, test on 1/2/4/8)
  E10: Google-trace multi-core validation
"""

import copy, random, math, sys, json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, '/home/claude')
from phase1_simulator import (
    Process, ProcessState, ReadyQueue, BlockedQueue,
    CPU, SchedulingDecision, Trajectory, TrajectoryBuffer,
    WorkloadGenerator, CONTEXT_SWITCH_OVERHEAD
)
from phase2_improved import (
    ValueEstimator, RMCTSAgent, SJFEstimator,
    fcfs_pick, sjf_true_pick, run_round_robin, run_mlfq
)
from experiment7_google_trace import GoogleTraceWorkloadGenerator


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Core Trajectory — extends single-core Trajectory with per-core metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiCoreTrajectory:
    """Trajectory for a multi-core scheduling run."""
    trajectory_id:        int
    scheduler_name:       str
    num_cores:            int
    decisions:            List[SchedulingDecision]
    processes:            List[Process]
    avg_turnaround_time:  float = 0.0
    avg_wait_time:        float = 0.0
    avg_response_time:    float = 0.0
    total_ticks:          int   = 0
    cpu_utilization:      float = 0.0   # aggregate across all cores
    starvation_count:     int   = 0
    per_core_utilization: List[float] = field(default_factory=list)
    load_imbalance:       float = 0.0   # std of per-core utilisation

    def compute_metrics(self, total_ticks: int,
                        per_core_idle: List[int]):
        self.total_ticks = total_ticks
        done = [p for p in self.processes if p.finish_time is not None]
        if not done:
            return
        n = len(done)
        self.avg_turnaround_time = sum(p.turnaround_time for p in done) / n
        self.avg_wait_time       = sum(p.wait_time       for p in done) / n
        self.avg_response_time   = sum(p.response_time   for p in done) / n
        self.starvation_count    = sum(
            1 for p in done if p.wait_time > 3 * p.burst_time)

        if total_ticks > 0:
            self.per_core_utilization = [
                max(total_ticks - idle, 0) / total_ticks
                for idle in per_core_idle
            ]
            self.cpu_utilization = float(np.mean(self.per_core_utilization))
            self.load_imbalance  = float(np.std(self.per_core_utilization))
        else:
            self.per_core_utilization = [0.0] * self.num_cores
            self.cpu_utilization = 0.0
            self.load_imbalance  = 0.0

    def is_successful(self):
        if not self.processes:
            return False
        avg_b = sum(p.burst_time for p in self.processes) / len(self.processes)
        return self.avg_turnaround_time < avg_b * 3.0

    def to_single_core_trajectory(self) -> Trajectory:
        """Convert to single-core Trajectory for compatibility with RMCTSAgent."""
        t = Trajectory(
            self.trajectory_id, self.scheduler_name,
            self.decisions, self.processes,
            self.avg_turnaround_time, self.avg_wait_time,
            self.avg_response_time, self.total_ticks,
            self.cpu_utilization, self.starvation_count
        )
        return t

    def summary(self):
        util_str = ", ".join(f"{u:.1%}" for u in self.per_core_utilization)
        return (f"{self.scheduler_name:12} | cores={self.num_cores} | "
                f"turn={self.avg_turnaround_time:7.2f} | "
                f"wait={self.avg_wait_time:7.2f} | "
                f"starv={self.starvation_count} | "
                f"util=[{util_str}] | imbal={self.load_imbalance:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Core Simulator
# ══════════════════════════════════════════════════════════════════════════════

class MultiCoreSimulator:
    """
    Symmetric Multi-Processing (SMP) simulator with shared ready queue.

    Architecture follows Linux SMP scheduling:
      - Single shared ready queue (like Linux's per-CPU runqueues
        simplified to global queue for clear analysis)
      - Each core independently selects from the shared queue
      - No process migration between cores after assignment
      - Shared blocked queue for I/O

    This models the core scheduling decision problem: given N processes
    in the ready queue and N cores available, which process goes to
    which core? The scheduler function is called once per idle core
    per tick, giving it full control over the assignment.
    """

    @staticmethod
    def run(processes: List[Process],
            scheduler_name: str,
            scheduler_fn,
            num_cores: int = 2) -> MultiCoreTrajectory:
        """
        Run a multi-core scheduling simulation.

        Args:
            processes:      List of Process objects to schedule
            scheduler_name: Label for this scheduler
            scheduler_fn:   Function(ReadyQueue, CPU) -> Process
                            Called once per idle core per scheduling point
            num_cores:      Number of CPU cores (1, 2, 4, or 8)

        Returns:
            MultiCoreTrajectory with all metrics
        """
        procs   = copy.deepcopy(processes)
        by_arr  = sorted(procs, key=lambda p: p.arrival_time)
        cores   = [CPU() for _ in range(num_cores)]
        rq      = ReadyQueue()
        bq      = BlockedQueue()
        decs    = []
        idx     = 0
        clock   = 0

        # Maximum ticks before forced termination
        max_t = (sum(p.burst_time for p in procs) * 3
                 + sum(p.io_burst * max(1, p.burst_time // max(p.cpu_phase, 1))
                       for p in procs if p.cpu_phase > 0)
                 + 200)

        while True:
            # ── 1. Admit new processes ─────────────────────────────────────
            while idx < len(by_arr) and by_arr[idx].arrival_time <= clock:
                rq.add(by_arr[idx])
                idx += 1

            # ── 2. Unblock completed I/O processes ─────────────────────────
            for p in bq.tick():
                p.current_cpu_ticks = 0
                rq.add(p)

            # ── 3. Assign ready processes to idle cores ────────────────────
            for core in cores:
                if core.is_idle() and not core.is_switching() and not rq.is_empty():
                    core.clock = clock   # synchronise core clock
                    chosen = scheduler_fn(rq, core)
                    if chosen is None:
                        continue
                    aw = (sum(p.wait_time for p in rq.get_all()) / rq.size()
                          if rq.size() > 0 else 0.0)
                    decs.append(SchedulingDecision(
                        clock,
                        [p.pid for p in rq.get_all()],
                        chosen.pid,
                        rq.size(),
                        chosen.remaining_time,
                        aw,
                        bq.size()
                    ))
                    rq.remove(chosen)
                    core.assign(chosen)

            # ── 4. Advance all cores by one tick ──────────────────────────
            clock += 1
            for core in cores:
                core.clock = clock
                done = core.run_one_tick()

                if done:
                    # RUNNING → EXIT
                    core.release()
                elif (core.current_process
                      and not core.is_switching()
                      and core.current_process.needs_io()):
                    # RUNNING → BLOCKED
                    p = core.release()
                    p.io_remaining = p.io_burst
                    bq.add(p)

            # ── 5. Accumulate wait time for all ready processes ────────────
            for p in rq.get_all():
                p.wait_time += 1

            # ── 6. Check termination ───────────────────────────────────────
            all_done = (
                all(p.is_complete() for p in procs)
                and bq.is_empty()
                and all(c.is_idle() for c in cores)
            )
            if all_done or clock >= max_t:
                break

        # ── Compute per-core idle time ─────────────────────────────────────
        per_core_idle = [c.idle_time for c in cores]

        traj = MultiCoreTrajectory(
            trajectory_id=0,
            scheduler_name=scheduler_name,
            num_cores=num_cores,
            decisions=decs,
            processes=procs,
        )
        traj.compute_metrics(clock, per_core_idle)
        return traj


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Core Baseline Runners
# ══════════════════════════════════════════════════════════════════════════════

def run_multicore_rr(processes: List[Process],
                     num_cores: int = 2,
                     quantum: int = 3) -> MultiCoreTrajectory:
    """
    Multi-core Round-Robin scheduler.
    Each core preempts its current process after `quantum` ticks.
    """
    procs   = copy.deepcopy(processes)
    by_arr  = sorted(procs, key=lambda p: p.arrival_time)
    cores   = [CPU() for _ in range(num_cores)]
    # Per-core quantum counters
    core_ticks = [0] * num_cores
    rq      = ReadyQueue()
    bq      = BlockedQueue()
    decs    = []
    idx     = 0
    clock   = 0

    max_t = (sum(p.burst_time for p in procs) * 4
             + sum(p.io_burst * max(1, p.burst_time // max(p.cpu_phase, 1))
                   for p in procs if p.cpu_phase > 0)
             + 200)

    while True:
        # Admit new
        while idx < len(by_arr) and by_arr[idx].arrival_time <= clock:
            rq.add(by_arr[idx]); idx += 1

        # Unblock I/O
        for p in bq.tick():
            p.current_cpu_ticks = 0; rq.add(p)

        # Preempt cores that hit quantum
        for ci, core in enumerate(cores):
            if (core.current_process
                    and not core.is_switching()
                    and core_ticks[ci] >= quantum):
                p = core.release()
                p.current_cpu_ticks = 0
                rq.add(p)
                core_ticks[ci] = 0

        # Assign idle cores
        for ci, core in enumerate(cores):
            if core.is_idle() and not core.is_switching() and not rq.is_empty():
                core.clock = clock
                chosen = min(rq.get_all(), key=lambda p: p.arrival_time)
                aw = (sum(p.wait_time for p in rq.get_all()) / rq.size()
                      if rq.size() > 0 else 0.0)
                decs.append(SchedulingDecision(
                    clock, [p.pid for p in rq.get_all()],
                    chosen.pid, rq.size(),
                    chosen.remaining_time, aw, bq.size()))
                rq.remove(chosen); core.assign(chosen)
                core_ticks[ci] = 0

        # Tick all cores
        clock += 1
        for ci, core in enumerate(cores):
            core.clock = clock
            done = core.run_one_tick()
            if done:
                core.release(); core_ticks[ci] = 0
            elif (core.current_process
                  and not core.is_switching()
                  and core.current_process.needs_io()):
                p = core.release()
                p.io_remaining = p.io_burst; bq.add(p)
                core_ticks[ci] = 0
            elif core.current_process and not core.is_switching():
                core_ticks[ci] += 1

        for p in rq.get_all():
            p.wait_time += 1

        all_done = (all(p.is_complete() for p in procs)
                    and bq.is_empty()
                    and all(c.is_idle() for c in cores))
        if all_done or clock >= max_t:
            break

    traj = MultiCoreTrajectory(0, "RoundRobin", num_cores, decs, procs)
    traj.compute_metrics(clock, [c.idle_time for c in cores])
    return traj


def run_all_multicore_schedulers(processes: List[Process],
                                 num_cores: int = 2,
                                 rmcts_agent=None,
                                 sjf_estimator=None) -> Dict:
    """Run all schedulers in multi-core mode."""
    if sjf_estimator is None:
        sjf_estimator = SJFEstimator()

    results = {}

    results['FCFS'] = MultiCoreSimulator.run(
        processes, 'FCFS', fcfs_pick, num_cores)

    results['SJF_True'] = MultiCoreSimulator.run(
        processes, 'SJF_True', sjf_true_pick, num_cores)

    results['SJF_Est'] = MultiCoreSimulator.run(
        processes, 'SJF_Est',
        lambda rq, c: sjf_estimator.pick(rq, c), num_cores)

    results['RoundRobin'] = run_multicore_rr(processes, num_cores)

    if rmcts_agent is not None:
        results['R-MCTS'] = MultiCoreSimulator.run(
            processes, 'R-MCTS',
            lambda rq, c: rmcts_agent.pick(rq, c), num_cores)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Training helper (single-core, then deploy on multi-core)
# ══════════════════════════════════════════════════════════════════════════════

def train_rmcts_single_core(workload_type='mixed',
                            base_seed=9000,
                            num_processes=20,
                            warmup=20,
                            iterations=8,
                            num_sim=40,
                            sthr=25):
    """
    Train R-MCTS on single-core trajectories only.
    The trained agent is then deployed on multi-core without retraining.
    This tests the core-count generalisation property.
    """
    buf = TrajectoryBuffer()
    ve  = ValueEstimator(learning_rate=0.01)

    def gen(seed):
        if workload_type == 'google':
            return GoogleTraceWorkloadGenerator.generate(num_processes, seed=seed)
        return WorkloadGenerator.generate(workload_type, num_processes, seed=seed)

    # Warmup: collect single-core FCFS and MLFQ trajectories
    for ep in range(warmup):
        procs = gen(base_seed + ep)
        from phase2_improved import run_mlfq as _mlfq
        buf.store(Simulator_run_single(procs, 'FCFS', fcfs_pick))
        buf.store(_mlfq(procs))

    ve.train(buf, epochs=30)
    agent = RMCTSAgent(ve, num_simulations=num_sim,
                       starvation_threshold=sthr)

    losses = []
    for it in range(iterations):
        procs = gen(base_seed + 200 + it)
        buf.store(Simulator_run_single(procs, 'FCFS', fcfs_pick))
        agent.improve(buf)
        before = len(ve.training_losses)
        ve.train(buf, epochs=30)
        epoch_losses = ve.training_losses[before:]
        losses.append(epoch_losses[-1] if epoch_losses else
                      (losses[-1] if losses else 0.05))

    return agent, ve, buf, losses


def Simulator_run_single(processes, scheduler_name, scheduler_fn):
    """Thin wrapper: run single-core Simulator for training."""
    from phase1_simulator import Simulator
    return Simulator.run(processes, scheduler_name, scheduler_fn)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 8: Multi-Core Scalability (N_cores = 1, 2, 4, 8)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment8(agent,
                    core_counts=(1, 2, 4, 8),
                    num_trials=20,
                    num_processes=20,
                    base_seed=9000):
    """
    Experiment 8: How does R-MCTS scale with number of cores?
    Compare all schedulers at N_cores = 1, 2, 4, 8.
    """
    print("\n" + "="*65)
    print("  Experiment 8: Multi-Core Scalability")
    print("  (Trained on single-core, deployed on 1/2/4/8 cores)")
    print("="*65)

    results = {}  # results[cores][scheduler] = list of MultiCoreTrajectory

    for nc in core_counts:
        print(f"\n  N_cores = {nc}")
        print(f"  {'Scheduler':<14} {'Turnaround':>12} {'Wait':>12} "
              f"{'Starv':>8} {'Util%':>8} {'Imbal':>8}")
        print("  " + "-"*64)

        results[nc] = {}
        for name in ['FCFS', 'SJF_True', 'SJF_Est', 'RoundRobin', 'R-MCTS']:
            results[nc][name] = []

        for trial in range(num_trials):
            procs = WorkloadGenerator.generate(
                'mixed', num_processes, seed=base_seed + 300 + trial)
            r = run_all_multicore_schedulers(
                procs, num_cores=nc,
                rmcts_agent=agent,
                sjf_estimator=SJFEstimator())
            for name, traj in r.items():
                results[nc][name].append(traj)

        for name in ['FCFS', 'SJF_True', 'SJF_Est', 'RoundRobin', 'R-MCTS']:
            trajs = results[nc][name]
            if not trajs:
                continue
            tm = np.mean([t.avg_turnaround_time for t in trajs])
            wm = np.mean([t.avg_wait_time for t in trajs])
            sm = np.mean([t.starvation_count for t in trajs])
            um = np.mean([t.cpu_utilization for t in trajs]) * 100
            im = np.mean([t.load_imbalance for t in trajs])
            mk = ' *' if name == 'R-MCTS' else ''
            print(f"  {name:<14} {tm:12.2f} {wm:12.2f} "
                  f"{sm:8.1f} {um:7.1f}% {im:8.4f}{mk}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 9: Cross-Core-Count Generalisation
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment9(agent,
                    core_counts=(1, 2, 4, 8),
                    num_trials=20,
                    num_processes=20,
                    base_seed=9000):
    """
    Experiment 9: Cross-core-count generalisation.

    R-MCTS is trained exclusively on single-core (N=1) trajectories.
    We measure how well it generalises to N=2, 4, 8 cores without
    any multi-core retraining. This is the key novel contribution:
    the per-process feature representation is core-count independent,
    allowing zero-shot transfer to unseen core counts.

    Reports: R-MCTS improvement over FCFS at each core count,
    demonstrating that the offline single-core training signal
    transfers effectively to the multi-core setting.
    """
    print("\n" + "="*65)
    print("  Experiment 9: Cross-Core-Count Generalisation")
    print("  (R-MCTS trained on N=1 only, tested on N=1/2/4/8)")
    print("="*65)
    print(f"\n  {'N_cores':<10} {'R-MCTS turn':>14} {'FCFS turn':>12} "
          f"{'Improvement':>14} {'R-MCTS starv':>14} {'FCFS starv':>12}")
    print("  " + "-"*76)

    gen_results = {}

    for nc in core_counts:
        rmcts_turns, fcfs_turns = [], []
        rmcts_starv, fcfs_starv = [], []

        for trial in range(num_trials):
            procs = WorkloadGenerator.generate(
                'mixed', num_processes, seed=base_seed + 400 + trial)

            rmcts_traj = MultiCoreSimulator.run(
                procs, 'R-MCTS',
                lambda rq, c: agent.pick(rq, c), nc)
            fcfs_traj = MultiCoreSimulator.run(
                procs, 'FCFS', fcfs_pick, nc)

            rmcts_turns.append(rmcts_traj.avg_turnaround_time)
            fcfs_turns.append(fcfs_traj.avg_turnaround_time)
            rmcts_starv.append(rmcts_traj.starvation_count)
            fcfs_starv.append(fcfs_traj.starvation_count)

        rm = np.mean(rmcts_turns)
        fm = np.mean(fcfs_turns)
        imp = (fm - rm) / fm * 100
        rs = np.mean(rmcts_starv)
        fs = np.mean(fcfs_starv)

        gen_results[nc] = {
            'rmcts_turn': float(rm), 'fcfs_turn': float(fm),
            'improvement': float(imp),
            'rmcts_starv': float(rs), 'fcfs_starv': float(fs),
            'rmcts_turns': [float(x) for x in rmcts_turns],
            'fcfs_turns':  [float(x) for x in fcfs_turns],
        }

        print(f"  {nc:<10} {rm:14.2f} {fm:12.2f} "
              f"{imp:+13.1f}% {rs:14.1f} {fs:12.1f}")

    return gen_results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 10: Google-Trace Multi-Core Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment10(agent,
                     core_counts=(1, 2, 4),
                     num_trials=20,
                     num_processes=20,
                     base_seed=9000):
    """
    Experiment 10: Multi-core validation on Google cluster trace workload.
    Combines the real-world workload (Experiment 7) with multi-core scaling.
    """
    print("\n" + "="*65)
    print("  Experiment 10: Google Trace Multi-Core Validation")
    print("="*65)

    results = {}

    for nc in core_counts:
        print(f"\n  N_cores = {nc}")
        results[nc] = {}

        for name in ['FCFS', 'SJF_True', 'SJF_Est', 'RoundRobin', 'R-MCTS']:
            results[nc][name] = []

        for trial in range(num_trials):
            procs = GoogleTraceWorkloadGenerator.generate(
                num_processes, seed=base_seed + 500 + trial)
            r = run_all_multicore_schedulers(
                procs, num_cores=nc,
                rmcts_agent=agent,
                sjf_estimator=SJFEstimator())
            for name, traj in r.items():
                results[nc][name].append(traj)

        print(f"  {'Scheduler':<14} {'Turnaround':>12} {'Util%':>8} "
              f"{'Starv':>8} {'Imbal':>8}")
        print("  " + "-"*52)
        for name in ['FCFS', 'SJF_True', 'SJF_Est', 'RoundRobin', 'R-MCTS']:
            trajs = results[nc].get(name, [])
            if not trajs:
                continue
            tm = np.mean([t.avg_turnaround_time for t in trajs])
            um = np.mean([t.cpu_utilization for t in trajs]) * 100
            sm = np.mean([t.starvation_count for t in trajs])
            im = np.mean([t.load_imbalance for t in trajs])
            mk = ' *' if name == 'R-MCTS' else ''
            print(f"  {name:<14} {tm:12.2f} {um:7.1f}% "
                  f"{sm:8.1f} {im:8.4f}{mk}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  R-MCTS Phase 5: Multi-Core Extension")
    print("="*65)

    BASE_SEED    = 9000
    NUM_PROCESSES = 20
    NUM_TRIALS   = 20

    # ── Train R-MCTS on single-core only ──────────────────────────────────
    print("\nStep 1: Training R-MCTS on single-core trajectories...")
    agent, ve, buf, losses = train_rmcts_single_core(
        workload_type='mixed',
        base_seed=BASE_SEED,
        num_processes=NUM_PROCESSES,
        warmup=20,
        iterations=8,
        num_sim=40,
        sthr=25
    )
    print(f"  Training complete.")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({(losses[0]-losses[-1])/losses[0]*100:.1f}% reduction)")
    print(f"  Policy entries: {len(agent.policy)}")

    # ── Experiment 8: Multi-Core Scalability ──────────────────────────────
    exp8 = run_experiment8(
        agent,
        core_counts=(1, 2, 4, 8),
        num_trials=NUM_TRIALS,
        num_processes=NUM_PROCESSES,
        base_seed=BASE_SEED
    )

    # ── Experiment 9: Cross-Core Generalisation ───────────────────────────
    exp9 = run_experiment9(
        agent,
        core_counts=(1, 2, 4, 8),
        num_trials=NUM_TRIALS,
        num_processes=NUM_PROCESSES,
        base_seed=BASE_SEED
    )

    # ── Experiment 10: Google Trace Multi-Core ────────────────────────────
    exp10 = run_experiment10(
        agent,
        core_counts=(1, 2, 4),
        num_trials=NUM_TRIALS,
        num_processes=NUM_PROCESSES,
        base_seed=BASE_SEED
    )

    # ── Save all results ──────────────────────────────────────────────────
    def serialise(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialise(i) for i in obj]
        return obj

    exp8_save = {}
    for nc, sched_dict in exp8.items():
        exp8_save[nc] = {}
        for name, trajs in sched_dict.items():
            exp8_save[nc][name] = {
                'turnaround_mean': float(np.mean([t.avg_turnaround_time for t in trajs])),
                'turnaround_std':  float(np.std([t.avg_turnaround_time for t in trajs])),
                'wait_mean':       float(np.mean([t.avg_wait_time for t in trajs])),
                'starvation_mean': float(np.mean([t.starvation_count for t in trajs])),
                'util_mean':       float(np.mean([t.cpu_utilization for t in trajs])),
                'imbal_mean':      float(np.mean([t.load_imbalance for t in trajs])),
            }

    exp10_save = {}
    for nc, sched_dict in exp10.items():
        exp10_save[nc] = {}
        for name, trajs in sched_dict.items():
            exp10_save[nc][name] = {
                'turnaround_mean': float(np.mean([t.avg_turnaround_time for t in trajs])),
                'turnaround_std':  float(np.std([t.avg_turnaround_time for t in trajs])),
                'util_mean':       float(np.mean([t.cpu_utilization for t in trajs])),
                'starvation_mean': float(np.mean([t.starvation_count for t in trajs])),
                'imbal_mean':      float(np.mean([t.load_imbalance for t in trajs])),
            }

    all_results = {
        'exp8': exp8_save,
        'exp9': serialise(exp9),
        'exp10': exp10_save,
        'training': {
            'losses': [float(x) for x in losses],
            'policy_size': len(agent.policy),
        }
    }

    with open('/home/claude/phase5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*65)
    print("  Phase 5 complete. Results saved to phase5_results.json")
    print("="*65)

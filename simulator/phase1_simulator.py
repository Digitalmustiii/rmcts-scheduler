"""
R-MCTS CPU Scheduler - Phase 1: Core Simulator
Author     : Sanusi Mustapha Babansoro  |  ID: 202524080114
Course     : Operating Systems: Structure and Applications, UESTC

Textbook alignment (Stallings, 9th Ed.):
  Five-State Process Model   -- Fig. 3.6
  Process Control Block      -- Table 3.4 / 3.5
  Context Switch Overhead    -- Table 3.7
  CPU-I/O Burst Cycle        -- Fig. 2.4 / 2.5
  Ready / Blocked Queues     -- Fig. 3.8
"""

import random, copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

CONTEXT_SWITCH_OVERHEAD = 1   # ticks lost per switch (Stallings Table 3.7)

# ── Five-State Model (Stallings Fig. 3.6) ─────────────────────────────────────
class ProcessState(Enum):
    NEW     = "new"
    READY   = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    EXIT    = "exit"

# ── Process Control Block (Stallings Table 3.4 / 3.5) ─────────────────────────
@dataclass
class Process:
    pid:          int
    arrival_time: int
    burst_time:   int          # total CPU time needed
    priority:     int = 1      # 1=highest, 5=lowest
    io_burst:     int = 0      # I/O ticks per blocking phase (Stallings Fig. 2.4)
    cpu_phase:    int = 0      # CPU ticks before I/O (0 = no blocking)

    # Runtime state (Stallings Table 3.5 -- Processor State Information)
    remaining_time:    int           = field(init=False)
    io_remaining:      int           = field(init=False, default=0)
    current_cpu_ticks: int           = field(init=False, default=0)
    state:             ProcessState  = field(init=False, default=ProcessState.NEW)
    start_time:        Optional[int] = field(init=False, default=None)
    finish_time:       Optional[int] = field(init=False, default=None)
    wait_time:         int           = field(init=False, default=0)
    context_switches:  int           = field(init=False, default=0)

    def __post_init__(self):
        self.remaining_time = self.burst_time

    @property
    def turnaround_time(self):
        return 0 if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def response_time(self):
        return 0 if self.start_time is None else self.start_time - self.arrival_time

    def is_complete(self):
        return self.remaining_time <= 0

    def needs_io(self):
        return (self.cpu_phase > 0 and self.io_burst > 0
                and self.current_cpu_ticks >= self.cpu_phase
                and self.remaining_time > 0)

    def __repr__(self):
        return (f"P{self.pid}(burst={self.burst_time}, cp={self.cpu_phase}, "
                f"io={self.io_burst}, arr={self.arrival_time}, pri={self.priority})")

# ── Ready Queue (Stallings Fig. 3.8) ──────────────────────────────────────────
class ReadyQueue:
    def __init__(self):               self._q: List[Process] = []
    def add(self, p):                 p.state = ProcessState.READY; self._q.append(p)
    def remove(self, p):              self._q.remove(p)
    def get_all(self):                return list(self._q)
    def is_empty(self):               return len(self._q) == 0
    def size(self):                   return len(self._q)

# ── Blocked Queue (Stallings Fig. 3.8) ────────────────────────────────────────
class BlockedQueue:
    def __init__(self):               self._q: List[Process] = []
    def add(self, p):                 p.state = ProcessState.BLOCKED; self._q.append(p)
    def is_empty(self):               return len(self._q) == 0
    def size(self):                   return len(self._q)

    def tick(self) -> List[Process]:
        """Decrement io_remaining for each blocked process.
        Returns processes whose I/O just completed (BLOCKED -> READY)."""
        done = [p for p in self._q if p.io_remaining - 1 <= 0]
        for p in self._q:
            p.io_remaining -= 1
        self._q[:] = [p for p in self._q if p.io_remaining > 0]
        for p in done:
            p.io_remaining = 0
        return done

# ── CPU ────────────────────────────────────────────────────────────────────────
class CPU:
    def __init__(self):
        self.clock   = 0
        self.current_process: Optional[Process] = None
        self.idle_time = 0
        self.total_context_switches = 0
        self._switching = False
        self._switch_ticks = 0
        self._last_pid: Optional[int] = None

    def assign(self, p: Process):
        if self._last_pid is not None and self._last_pid != p.pid:
            self._switching = True
            self._switch_ticks = CONTEXT_SWITCH_OVERHEAD
            self.total_context_switches += 1
            p.context_switches += 1
        self.current_process = p
        p.state = ProcessState.RUNNING
        if p.start_time is None:
            p.start_time = self.clock

    def run_one_tick(self) -> bool:
        self.clock += 1
        if self._switching:
            self.idle_time += 1
            self._switch_ticks -= 1
            if self._switch_ticks <= 0:
                self._switching = False
            return False
        if self.current_process:
            self.current_process.remaining_time    -= 1
            self.current_process.current_cpu_ticks += 1
            if self.current_process.is_complete():
                self.current_process.finish_time = self.clock
                self.current_process.state = ProcessState.EXIT
                return True
        else:
            self.idle_time += 1
        return False

    def release(self) -> Optional[Process]:
        p = self.current_process
        if p:
            self._last_pid = p.pid
        self.current_process = None
        return p

    def is_idle(self):      return self.current_process is None and not self._switching
    def is_switching(self): return self._switching

# ── Scheduling Decision ────────────────────────────────────────────────────────
@dataclass
class SchedulingDecision:
    clock:                int
    queue_snapshot:       List[int]
    chosen_pid:           int
    queue_lengths:        int
    chosen_remaining:     int
    avg_wait_at_decision: float
    blocked_count:        int = 0

# ── Trajectory ─────────────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    trajectory_id:       int
    scheduler_name:      str
    decisions:           List[SchedulingDecision]
    processes:           List[Process]
    avg_turnaround_time: float = 0.0
    avg_wait_time:       float = 0.0
    avg_response_time:   float = 0.0
    total_ticks:         int   = 0
    cpu_utilization:     float = 0.0
    starvation_count:    int   = 0

    def compute_metrics(self, total_ticks, idle_ticks):
        self.total_ticks = total_ticks
        done = [p for p in self.processes if p.finish_time is not None]
        if not done:
            return
        n = len(done)
        self.avg_turnaround_time = sum(p.turnaround_time for p in done) / n
        self.avg_wait_time       = sum(p.wait_time       for p in done) / n
        self.avg_response_time   = sum(p.response_time   for p in done) / n
        self.cpu_utilization     = max(total_ticks-idle_ticks,0)/total_ticks if total_ticks else 0.0
        self.starvation_count    = sum(1 for p in done if p.wait_time > 3*p.burst_time)

    def is_successful(self):
        if not self.processes:
            return False
        avg_b = sum(p.burst_time for p in self.processes) / len(self.processes)
        return self.avg_turnaround_time < avg_b * 3.0

    def summary(self):
        return (f"{self.scheduler_name:12} | turn={self.avg_turnaround_time:6.2f} | "
                f"wait={self.avg_wait_time:6.2f} | resp={self.avg_response_time:5.2f} | "
                f"util={self.cpu_utilization:.1%} | starv={self.starvation_count}")

# ── Trajectory Buffer ──────────────────────────────────────────────────────────
class TrajectoryBuffer:
    def __init__(self):
        self._buf: List[Trajectory] = []
        self._id = 0

    def store(self, t):
        t.trajectory_id = self._id; self._id += 1; self._buf.append(t)

    def get_all(self):        return list(self._buf)
    def get_failed(self):     return [t for t in self._buf if not t.is_successful()]
    def get_successful(self): return [t for t in self._buf if t.is_successful()]
    def size(self):           return len(self._buf)

    def average_turnaround(self):
        return sum(t.avg_turnaround_time for t in self._buf)/len(self._buf) if self._buf else 0.0

    def stats_summary(self):
        s = len(self.get_successful())
        return (f"Buffer: {self.size()} | Successful: {s} | Failed: {self.size()-s} | "
                f"Avg Turnaround: {self.average_turnaround():.2f}")

# ── Workload Generator ─────────────────────────────────────────────────────────
class WorkloadGenerator:
    @staticmethod
    def generate(workload_type="mixed", num_processes=8, seed=None):
        if seed is not None:
            random.seed(seed)
        procs = []
        for i in range(num_processes):
            arr, pri = random.randint(0, num_processes*2), random.randint(1, 5)
            if workload_type == "cpu_bound":
                cp, io, n = random.randint(8,20), random.randint(1,3), random.randint(1,2)
            elif workload_type == "io_bound":
                cp, io, n = random.randint(2,5), random.randint(5,14), random.randint(2,5)
            else:
                if random.random() < 0.5:
                    cp, io, n = random.randint(5,18), random.randint(1,5), random.randint(1,2)
                else:
                    cp, io, n = random.randint(2,6), random.randint(4,12), random.randint(2,4)
            procs.append(Process(pid=i+1, arrival_time=arr, burst_time=cp*n,
                                 priority=pri, io_burst=io, cpu_phase=cp))
        procs.sort(key=lambda p: p.arrival_time)
        return procs

# ── Simulator ──────────────────────────────────────────────────────────────────
class Simulator:
    @staticmethod
    def run(processes, scheduler_name, scheduler_fn):
        procs  = copy.deepcopy(processes)
        by_arr = sorted(procs, key=lambda p: p.arrival_time)
        cpu    = CPU()
        rq     = ReadyQueue()
        bq     = BlockedQueue()
        decs   = []
        idx    = 0
        max_t  = (sum(p.burst_time for p in procs)*3
                  + sum(p.io_burst*max(1, p.burst_time//max(p.cpu_phase,1)) for p in procs)
                  + 100)

        while True:
            while idx < len(by_arr) and by_arr[idx].arrival_time <= cpu.clock:
                rq.add(by_arr[idx]); idx += 1              # NEW -> READY

            for p in bq.tick():                            # BLOCKED -> READY
                p.current_cpu_ticks = 0; rq.add(p)

            if cpu.is_idle() and not rq.is_empty():        # READY -> RUNNING
                chosen = scheduler_fn(rq, cpu)
                aw = sum(p.wait_time for p in rq.get_all()) / rq.size()
                decs.append(SchedulingDecision(cpu.clock,
                    [p.pid for p in rq.get_all()], chosen.pid,
                    rq.size(), chosen.remaining_time, aw, bq.size()))
                rq.remove(chosen); cpu.assign(chosen)

            done = cpu.run_one_tick()
            for p in rq.get_all():
                p.wait_time += 1

            if done:                                                   # RUNNING -> EXIT
                cpu.release()
            elif (cpu.current_process and not cpu.is_switching()
                  and cpu.current_process.needs_io()):                 # RUNNING -> BLOCKED
                p = cpu.release(); p.io_remaining = p.io_burst; bq.add(p)

            if (all(p.is_complete() for p in procs) and bq.is_empty()) or cpu.clock >= max_t:
                break

        t = Trajectory(0, scheduler_name, decs, procs)
        t.compute_metrics(cpu.clock, cpu.idle_time)
        return t

# ── I/O tick helper for preemptive schedulers in phase2 ───────────────────────
def advance_blocked(blocked: List[Process], ready: List[Process]):
    still = []
    for p in blocked:
        p.io_remaining -= 1
        if p.io_remaining <= 0:
            p.io_remaining = 0; p.current_cpu_ticks = 0
            p.state = ProcessState.READY; ready.append(p)
        else:
            still.append(p)
    blocked[:] = still

def fcfs_pick(rq, cpu):
    return min(rq.get_all(), key=lambda p: p.arrival_time)

if __name__ == "__main__":
    buf = TrajectoryBuffer()
    for ep in range(3):
        procs = WorkloadGenerator.generate("mixed", 6, seed=ep*42)
        t     = Simulator.run(procs, "FCFS", fcfs_pick)
        buf.store(t); print(f"Ep {ep+1}: {t.summary()}")
    print(buf.stats_summary())
    print(f"CS overhead={CONTEXT_SWITCH_OVERHEAD}  Phase 1 OK")
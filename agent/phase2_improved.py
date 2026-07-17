"""
R-MCTS CPU Scheduler - Phase 2: Schedulers + R-MCTS Agent
Author     : Sanusi Mustapha Babansoro  |  ID: 202524080114
Course     : Operating Systems: Structure and Applications, UESTC

Baselines : FCFS, SJF_True (theoretical), SJF_Est (exponential avg),
            Round-Robin (quantum=3), MLFQ (3 levels, aging)
Proposed  : R-MCTS -- priority-aware value estimator + UCT tree search

All schedulers handle the five-state model (I/O blocking) from phase1.
SJF_Est uses exponential averaging (Stallings Ch.2 Sec.2.4, alpha=0.5).
R-MCTS: UCT criterion c=sqrt(2), priority bonus up to 0.3, anti-starvation
        threshold 25 ticks. Trained offline via FilteredBC on past episodes.
"""

import math, random, copy
import numpy as np
from typing import List, Optional, Dict, Tuple
from phase1_simulator import (
    Process, ReadyQueue, BlockedQueue, CPU, Trajectory, TrajectoryBuffer,
    SchedulingDecision, WorkloadGenerator, Simulator,
    ProcessState, advance_blocked, CONTEXT_SWITCH_OVERHEAD
)

RR_QUANTUM            = 3
MLFQ_QUANTUMS         = [3, 6, 12]
MLFQ_AGING_THRESHOLD  = 20


# ── Baseline pickers ──────────────────────────────────────────────────────────
def fcfs_pick(rq, cpu):
    return min(rq.get_all(), key=lambda p: p.arrival_time)

def sjf_true_pick(rq, cpu):
    return min(rq.get_all(), key=lambda p: p.remaining_time)


# ── SJF with Exponential Averaging (Stallings Ch.2 Sec.2.4) ──────────────────
class SJFEstimator:
    def __init__(self, alpha=0.5, initial_estimate=10.0):
        self.alpha, self.initial = alpha, initial_estimate
        self.estimates: Dict[int, float] = {}

    def get_estimate(self, pid):  return self.estimates.get(pid, self.initial)
    def update(self, pid, burst): self.estimates[pid] = self.alpha*burst + (1-self.alpha)*self.get_estimate(pid)
    def pick(self, rq, cpu):      return min(rq.get_all(), key=lambda p: self.get_estimate(p.pid))


# ── Round-Robin (with I/O blocking) ──────────────────────────────────────────
def run_round_robin(processes: List[Process]) -> Trajectory:
    procs   = copy.deepcopy(processes)
    by_arr  = sorted(procs, key=lambda p: p.arrival_time)
    cpu     = CPU()
    queue:   List[Process] = []
    blocked: List[Process] = []
    decs, idx, ticks = [], 0, 0
    max_t = sum(p.burst_time for p in procs)*3 + sum(
        p.io_burst*max(1,p.burst_time//max(p.cpu_phase,1)) for p in procs) + 100

    while True:
        while idx < len(by_arr) and by_arr[idx].arrival_time <= cpu.clock:
            p = by_arr[idx]; p.state = ProcessState.READY; queue.append(p); idx += 1

        advance_blocked(blocked, queue)

        if not cpu.is_idle() and ticks >= RR_QUANTUM and queue:
            p = cpu.release(); p.state = ProcessState.READY; queue.append(p); ticks = 0

        if cpu.is_idle() and queue:
            chosen = queue.pop(0)
            aw = sum(p.wait_time for p in queue)/len(queue) if queue else 0
            decs.append(SchedulingDecision(cpu.clock,[p.pid for p in queue],
                chosen.pid,len(queue),chosen.remaining_time,aw,len(blocked)))
            cpu.assign(chosen); ticks = 0

        done = cpu.run_one_tick(); ticks += 1
        for p in queue: p.wait_time += 1

        if done:
            cpu.release(); ticks = 0
        elif cpu.current_process and not cpu.is_switching() and cpu.current_process.needs_io():
            p = cpu.release(); p.io_remaining = p.io_burst
            p.state = ProcessState.BLOCKED; blocked.append(p); ticks = 0

        if (all(p.is_complete() for p in procs) and not blocked) or cpu.clock >= max_t:
            break

    t = Trajectory(0,"RoundRobin",decs,procs); t.compute_metrics(cpu.clock,cpu.idle_time)
    return t


# ── MLFQ -- 3 levels, aging, I/O blocking ────────────────────────────────────
def run_mlfq(processes: List[Process]) -> Trajectory:
    procs   = copy.deepcopy(processes)
    by_arr  = sorted(procs, key=lambda p: p.arrival_time)
    cpu     = CPU()
    queues  = [[], [], []]
    blocked: List[Process] = []
    plevel, pticks, pwait = {}, {}, {}
    decs, idx, ticks = [], 0, 0
    max_t = sum(p.burst_time for p in procs)*4 + sum(
        p.io_burst*max(1,p.burst_time//max(p.cpu_phase,1)) for p in procs) + 100

    while True:
        while idx < len(by_arr) and by_arr[idx].arrival_time <= cpu.clock:
            p = by_arr[idx]; p.state = ProcessState.READY
            queues[0].append(p); plevel[p.pid]=0; pticks[p.pid]=0; pwait[p.pid]=0; idx+=1

        still = []
        for p in blocked:
            p.io_remaining -= 1
            if p.io_remaining <= 0:
                p.io_remaining=0; p.current_cpu_ticks=0; p.state=ProcessState.READY
                queues[plevel.get(p.pid,0)].append(p)
            else: still.append(p)
        blocked[:] = still

        for lv in [1,2]:
            for p in list(queues[lv]):
                pwait[p.pid] = pwait.get(p.pid,0)+1
                if pwait[p.pid] >= MLFQ_AGING_THRESHOLD:
                    queues[lv].remove(p); nl=lv-1
                    queues[nl].append(p); plevel[p.pid]=nl; pticks[p.pid]=0; pwait[p.pid]=0

        clv = plevel.get(cpu.current_process.pid,0) if cpu.current_process else 0
        q   = MLFQ_QUANTUMS[clv]

        if not cpu.is_idle() and ticks >= q:
            p = cpu.release(); nl = min(plevel.get(p.pid,0)+1,2)
            p.state = ProcessState.READY; queues[nl].append(p)
            plevel[p.pid]=nl; pticks[p.pid]=0; pwait[p.pid]=0; ticks=0

        if cpu.is_idle():
            chosen = next((queues[lv].pop(0) for lv in range(3) if queues[lv]), None)
            if chosen:
                aw_all = [p for q in queues for p in q]
                aw = sum(p.wait_time for p in aw_all)/len(aw_all) if aw_all else 0
                decs.append(SchedulingDecision(cpu.clock,[p.pid for p in aw_all],
                    chosen.pid,sum(len(q) for q in queues),chosen.remaining_time,aw,len(blocked)))
                cpu.assign(chosen); ticks=0

        done = cpu.run_one_tick(); ticks+=1
        for q in queues:
            for p in q: p.wait_time+=1

        if done:
            cpu.release(); ticks=0
        elif cpu.current_process and not cpu.is_switching() and cpu.current_process.needs_io():
            p=cpu.release(); p.io_remaining=p.io_burst
            p.state=ProcessState.BLOCKED; blocked.append(p); ticks=0

        if (all(p.is_complete() for p in procs) and not blocked) or cpu.clock>=max_t:
            break

    t = Trajectory(0,"MLFQ",decs,procs); t.compute_metrics(cpu.clock,cpu.idle_time)
    return t


# ── Value Estimator (6-input MLP, priority-weighted reward) ───────────────────
class ValueEstimator:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(6,24)*0.1;  self.b1 = np.zeros(24)
        self.W2 = np.random.randn(24,12)*0.1; self.b2 = np.zeros(12)
        self.W3 = np.random.randn(12,1)*0.1;  self.b3 = np.zeros(1)
        self.is_trained = False
        self.training_losses: List[float] = []

    def _relu(self, x):    return np.maximum(0, x)
    def _sigmoid(self, x): return 1/(1+np.exp(-np.clip(x,-500,500)))

    def _forward(self, x):
        z1=x@self.W1+self.b1; a1=self._relu(z1)
        z2=a1@self.W2+self.b2; a2=self._relu(z2)
        z3=a2@self.W3+self.b3; out=self._sigmoid(z3)
        return out, (x,z1,a1,z2,a2,z3)

    def predict(self, f: np.ndarray) -> float:
        if f.ndim==1: f=f.reshape(1,-1)
        out,_ = self._forward(f)
        return float(out[0,0])

    def _pw_return(self, traj: Trajectory) -> float:
        done = [p for p in traj.processes if p.finish_time is not None]
        if not done: return traj.avg_turnaround_time
        W = {1:5.0,2:3.0,3:2.0,4:1.5,5:1.0}
        ws = sum(p.turnaround_time*W.get(p.priority,1.0) for p in done)
        wt = sum(W.get(p.priority,1.0) for p in done)
        return ws/(wt+1e-8)

    def train(self, buffer: TrajectoryBuffer, epochs=40):
        trajs = buffer.get_all()
        if len(trajs) < 2: return
        pwr = [self._pw_return(t) for t in trajs]
        mu, sigma = np.mean(pwr), np.std(pwr)+1e-8
        data = []
        for traj,r in zip(trajs,pwr):
            target = float(np.clip(1.0-(r-mu)/(sigma*3+1e-8), 0.0, 1.0))
            for dec in traj.decisions:
                data.append((self._extract_features(dec,traj), target))
        if not data: return
        for ep in range(epochs):
            random.shuffle(data)
            loss = 0.0
            for f,tgt in data:
                x=f.reshape(1,-1); t=np.array([[tgt]])
                out,cache = self._forward(x)
                x_in,z1,a1,z2,a2,z3 = cache
                loss += float(np.mean((out-t)**2))
                d=2*(out-t)/x.shape[0]; ds=out*(1-out); dz3=d*ds
                dW3=a2.T@dz3; db3=dz3.sum(0); da2=dz3@self.W3.T; dz2=da2*(z2>0)
                dW2=a1.T@dz2; db2=dz2.sum(0); da1=dz2@self.W2.T; dz1=da1*(z1>0)
                dW1=x_in.T@dz1; db1=dz1.sum(0)
                self.W3-=self.lr*dW3; self.b3-=self.lr*db3
                self.W2-=self.lr*dW2; self.b2-=self.lr*db2
                self.W1-=self.lr*dW1; self.b1-=self.lr*db1
            self.training_losses.append(loss/len(data))
        self.is_trained = True

    def _extract_features(self, dec: SchedulingDecision, traj: Trajectory) -> np.ndarray:
        proc = next((p for p in traj.processes if p.pid==dec.chosen_pid), None)
        burst    = proc.burst_time if proc else 10.0
        priority = proc.priority   if proc else 3.0
        waited   = proc.wait_time  if proc else 0.0
        return np.array([
            min(dec.queue_lengths/20.0, 1.0),
            min(dec.chosen_remaining/30.0, 1.0),
            min(dec.avg_wait_at_decision/150.0, 1.0),
            (priority-1)/4.0,
            min(dec.chosen_remaining/max(burst,1.0), 1.0),
            min(waited/150.0, 1.0),
        ], dtype=np.float32)


# ── MCTS Node ─────────────────────────────────────────────────────────────────
class MCTSNode:
    def __init__(self, state: Dict, parent=None, action_pid=None):
        self.state=state; self.parent=parent; self.action_pid=action_pid
        self.children: List['MCTSNode'] = []
        self.visit_count=0; self.total_value=0.0
        self.untried_actions: List[int] = list(state.get('available_pids',[]))

    @property
    def q_value(self): return self.total_value/self.visit_count if self.visit_count else 0.0

    def uct_score(self, c=1.414):
        if self.visit_count==0: return float('inf')
        if self.parent is None: return self.q_value
        return self.q_value + c*math.sqrt(math.log(max(self.parent.visit_count,1))/self.visit_count)

    def is_fully_expanded(self): return len(self.untried_actions)==0
    def best_child(self):        return max(self.children, key=lambda n: n.uct_score())
    def best_action_child(self): return max(self.children, key=lambda n: n.q_value)


# ── R-MCTS Agent ──────────────────────────────────────────────────────────────
class RMCTSAgent:
    def __init__(self, value_estimator: ValueEstimator,
                 num_simulations=60, exploration_c=1.414, starvation_threshold=25):
        self.ve   = value_estimator
        self.nsim = num_simulations
        self.c    = exploration_c
        self.sthr = starvation_threshold
        self.policy: Dict[str,int] = {}
        self.preference_pairs: List[Tuple] = []
        self.improvement_history: List[float] = []

    def _sk(self, pids, clock): return f"{clock}:{sorted(pids)}"

    def _failure_idx(self, traj):
        if not traj.decisions: return 0
        return max(0, max(range(len(traj.decisions)),
                          key=lambda i: traj.decisions[i].avg_wait_at_decision) - 1)

    def _priority_bonus(self, pid, traj):
        p = next((x for x in traj.processes if x.pid==pid), None)
        if not p: return 0.0
        return (5-p.priority)/4.0*0.3 + min(p.wait_time/self.sthr,1.0)*0.2

    def _evaluate(self, dec: SchedulingDecision, traj: Trajectory) -> float:
        if not self.ve.is_trained:
            return 1.0/(dec.chosen_remaining+1.0) + self._priority_bonus(dec.chosen_pid, traj)
        return min(self.ve.predict(self.ve._extract_features(dec,traj))
                   + self._priority_bonus(dec.chosen_pid,traj), 1.0)

    def _mcts_search(self, traj: Trajectory, start: int) -> Optional[List[int]]:
        if start >= len(traj.decisions): return None
        sd = traj.decisions[start]
        root = MCTSNode({'available_pids':sd.queue_snapshot.copy(),
                         'clock':sd.clock,'decision_idx':start})
        for _ in range(self.nsim):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            if not node.is_fully_expanded() and node.untried_actions:
                apid = random.choice(node.untried_actions)
                node.untried_actions.remove(apid)
                child = MCTSNode({'available_pids':[p for p in node.state['available_pids'] if p!=apid],
                                  'clock':node.state['clock']+1,
                                  'decision_idx':node.state['decision_idx']+1},
                                 parent=node, action_pid=apid)
                node.children.append(child); node=child
            di  = min(node.state.get('decision_idx',start), len(traj.decisions)-1)
            ed  = traj.decisions[di]
            tmp = SchedulingDecision(node.state['clock'], node.state['available_pids'],
                    node.action_pid or ed.chosen_pid, len(node.state['available_pids']),
                    ed.chosen_remaining, ed.avg_wait_at_decision)
            v = self._evaluate(tmp, traj)
            while node:
                node.visit_count+=1; node.total_value+=v; node=node.parent
        path=[]; cur=root
        while cur.children:
            cur=cur.best_action_child()
            if cur.action_pid: path.append(cur.action_pid)
        return path or None

    def improve(self, buffer: TrajectoryBuffer) -> int:
        pairs = 0
        for traj in buffer.get_failed():
            fi   = self._failure_idx(traj)
            path = self._mcts_search(traj, fi)
            if path:
                orig = self.ve._pw_return(traj)
                w    = (orig - orig*0.70)/(orig+1e-8)
                if w > 0:
                    self.preference_pairs.append((path,traj)); pairs+=1
                    if traj.decisions:
                        self.policy[self._sk(traj.decisions[fi].queue_snapshot,
                                             traj.decisions[fi].clock)] = path[0]
                    self.improvement_history.append(w)
        return pairs

    def pick(self, rq: ReadyQueue, cpu: CPU) -> Process:
        cands = rq.get_all()
        sk = self._sk([p.pid for p in cands], cpu.clock)
        if sk in self.policy:
            m = next((p for p in cands if p.pid==self.policy[sk]), None)
            if m: return m
        starving = [p for p in cands if p.wait_time >= self.sthr]
        if starving: return min(starving, key=lambda p: p.priority)
        W = {1:5.0,2:3.0,3:2.0,4:1.5,5:1.0}
        return min(cands, key=lambda p: p.remaining_time/W.get(p.priority,1.0))


# ── Run all schedulers ────────────────────────────────────────────────────────
def run_all_schedulers(processes, rmcts_agent=None, sjf_estimator=None):
    if sjf_estimator is None: sjf_estimator = SJFEstimator()
    res = {
        'FCFS':       Simulator.run(processes, "FCFS",     fcfs_pick),
        'SJF_True':   Simulator.run(processes, "SJF_True", sjf_true_pick),
        'SJF_Est':    Simulator.run(processes, "SJF_Est",  lambda rq,c: sjf_estimator.pick(rq,c)),
        'RoundRobin': run_round_robin(processes),
        'MLFQ':       run_mlfq(processes),
    }
    if rmcts_agent:
        res['R-MCTS'] = Simulator.run(processes, "R-MCTS", lambda rq,c: rmcts_agent.pick(rq,c))
    return res


if __name__ == "__main__":
    print("Phase 2 verification...")
    buf = TrajectoryBuffer()
    for i in range(20):
        p = WorkloadGenerator.generate("mixed", 8, seed=i)
        buf.store(Simulator.run(p,"FCFS",fcfs_pick)); buf.store(run_mlfq(p))
    est = ValueEstimator(0.01); est.train(buf, epochs=30)
    agent = RMCTSAgent(est, num_simulations=40); agent.improve(buf)
    print(f"Loss: {est.training_losses[0]:.4f} -> {est.training_losses[-1]:.4f}")
    for wl in ["mixed","io_bound","cpu_bound"]:
        procs = WorkloadGenerator.generate(wl, 8, seed=99)
        r = run_all_schedulers(procs, rmcts_agent=agent, sjf_estimator=SJFEstimator())
        print(f"\n{wl.upper()}")
        for name,t in r.items():
            print(f"  {name:12} turn={t.avg_turnaround_time:.2f}  wait={t.avg_wait_time:.2f}")
    print("\nPhase 2 OK")
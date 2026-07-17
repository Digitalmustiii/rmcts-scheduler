# R-MCTS: Zero-Shot Multi-Core Process Scheduling via Offline Monte Carlo Tree Search with Contrastive Value Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Preprint](https://img.shields.io/badge/preprint-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.21317887)

## Overview

This repository contains the full implementation of **R-MCTS**, an offline Monte Carlo Tree Search scheduler that:

- Trains exclusively on **single-core** trajectories using contrastive offline value learning
- Deploys **without retraining** on SMP systems with 1, 2, 4, or 8 cores (zero-shot core-count transfer)
- Achieves turnaround within **1.9%** of the SJF-True theoretical ceiling on single-core workloads
- Reduces starvation-related short-process wait time by **90.0%** vs FCFS
- Achieves the **lowest load imbalance** (σ_U) at every multi-core configuration tested

## Paper

> Sanusi Mustapha Babansoro and Abrha Gebriye Embafresu.
> "Zero-Shot Multi-Core Process Scheduling via Offline Monte Carlo Tree Search with Contrastive Value Learning."
> School of Computer Science and Engineering, UESTC, Chengdu, China, 2026.
> Preprint: https://doi.org/10.5281/zenodo.21317887

## Repository Structure

```
rmcts/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── simulator/
│   ├── phase1_simulator.py            # Discrete-event OS simulator (single-core)
│   └── phase5_multicore.py            # SMP multi-core extension
│
├── agent/
│   ├── phase2_improved.py             # R-MCTS agent, value estimator, baselines
│   └── experiment7_google_trace.py    # Google cluster trace workload generator
│
├── experiments/
│   ├── phase3_experiments.py          # Experiments 1-6 (single-core)
│   ├── phase3_publication.py          # Publication-quality experiment runner
│   └── phase4_final.py               # Final experiment configurations
│
├── figures/
│   ├── generate_ieee_figures_pub.py   # Figure generation script (all 14 figures)
│   ├── fig1_learning_1col.png         # Exp 1: Learning curve
│   ├── fig2_comparison_2col.png       # Exp 2: Main benchmark
│   ├── fig3_starvation_2col.png       # Exp 3: Starvation stress test
│   ├── fig4_scalability_1col.png      # Exp 4: Single-core scalability
│   ├── fig5_ablation_2col.png         # Exp 5: Ablation study
│   ├── fig6_workloads_2col.png        # Exp 6: Cross-workload results
│   ├── fig7_response_2col.png         # Exp 6: Response time
│   ├── fig8_google_trace_2col.png     # Exp 7: Google trace validation
│   ├── fig9_threshold_sensitivity_1col.png  # Exp 7b: Threshold sensitivity
│   ├── fig10_convergence_google_1col.png    # Exp 7c: Training convergence
│   ├── fig11_cpu_utilisation_1col.png       # Exp 7d: CPU utilisation
│   ├── fig12_multicore_scalability_2col.png # Exp 8: Multi-core scalability
│   ├── fig13_generalisation_1col.png        # Exp 9: Core-count transfer
│   └── fig14_google_multicore_2col.png      # Exp 10: Google trace multi-core
│
└── paper/
    └── rmcts_tpds.tex                 # LaTeX source (IEEEtran format)
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/rmcts-scheduler.git
cd rmcts-scheduler
pip install -r requirements.txt
```

## Quick Start

### Run Single-Core Experiments (1-6)

```python
from simulator.phase1_simulator import WorkloadGenerator, Simulator
from agent.phase2_improved import RMCTSAgent, ValueEstimator, TrajectoryBuffer
from agent.phase2_improved import run_all_schedulers, fcfs_pick, run_mlfq

# Generate mixed workload
processes = WorkloadGenerator.generate('mixed', num_processes=20, seed=42)

# Train R-MCTS
buf = TrajectoryBuffer()
ve  = ValueEstimator(learning_rate=0.01)

# Warmup
for ep in range(30):
    procs = WorkloadGenerator.generate('mixed', 20, seed=ep)
    buf.store(Simulator.run(procs, 'FCFS', fcfs_pick))
    buf.store(run_mlfq(procs))

ve.train(buf, epochs=40)
agent = RMCTSAgent(ve, num_simulations=60, starvation_threshold=25)

# Offline improvement
for it in range(10):
    procs = WorkloadGenerator.generate('mixed', 20, seed=100+it)
    buf.store(Simulator.run(procs, 'FCFS', fcfs_pick))
    agent.improve(buf)
    ve.train(buf, epochs=40)

# Run all schedulers and compare
results = run_all_schedulers(processes, rmcts_agent=agent)
for name, traj in results.items():
    print(f"{name:<14}: turnaround={traj.avg_turnaround_time:.2f}  "
          f"wait={traj.avg_wait_time:.2f}  starvation={traj.starvation_count}")
```

### Run Multi-Core Experiments (8-10)

```python
from simulator.phase5_multicore import MultiCoreSimulator, run_all_multicore_schedulers

# Deploy trained agent on 4 cores without retraining
results = run_all_multicore_schedulers(
    processes,
    num_cores=4,
    rmcts_agent=agent  # Same agent trained on single-core only
)

for name, traj in results.items():
    print(f"{name:<14}: turnaround={traj.avg_turnaround_time:.2f}  "
          f"util={traj.cpu_utilization:.1%}  imbalance={traj.load_imbalance:.4f}")
```

### Google Cluster Trace Workload

```python
from agent.experiment7_google_trace import GoogleTraceWorkloadGenerator

# Generate processes with Google-trace-derived statistics
# (lognormal burst distribution, Poisson arrivals, Google priority mapping)
processes = GoogleTraceWorkloadGenerator.generate(num_processes=20, seed=42)
```

## Reproducing All Results

```bash
# Run all 10 experiments and regenerate all 14 figures
python experiments/phase3_publication.py    # Experiments 1-6
python agent/experiment7_google_trace.py    # Experiment 7
python simulator/phase5_multicore.py        # Experiments 8-10
python figures/generate_ieee_figures_pub.py # All figures
```

Expected runtime: approximately 280 minutes on a standard laptop (Windows/Mac/Linux).

## Key Results

| Experiment | Metric | R-MCTS | Best Baseline | Improvement |
|---|---|---|---|---|
| Exp 2: Mixed workload | Turnaround | 88.71 ticks | SJF-Est: 116.36 | +23.8% (p<0.001) |
| Exp 2: vs theoretical | Turnaround | 88.71 ticks | SJF-True: 87.03 | 1.9% gap (p=0.667) |
| Exp 3: Starvation | Short-proc wait | 5.54 ticks | FCFS: 55.64 | 90.0% reduction |
| Exp 7: Google trace | CPU utilisation | 90.8% | All others lower | Best of all |
| Exp 8: N=2 cores | Load imbalance σ_U | 0.024 | FCFS: 0.031 | Lowest |
| Exp 9: Transfer | vs FCFS at N=4 | +2.2% | — | Zero-shot transfer |

## Architecture

```
Single-core training (N=1)
         │
         ▼
   Trajectory Buffer
   (failed episodes)
         │
         ▼
   UCT Tree Search ──► Contrastive Value Estimator (V_θ)
         │                    6-input MLP 24-12-1
         ▼
   Policy Table (Π)
   (ANN lookup)
         │
         ▼
   Runtime Scheduler ──► Deploy on N=1,2,4,8 cores
   (zero-shot transfer)   WITHOUT retraining
```

## Schedulers Compared

| Scheduler | Type | Description |
|---|---|---|
| FCFS | Baseline | First-Come-First-Served |
| SJF-True | Ceiling | Exact burst times (unrealisable) |
| SJF-Est | Realistic | Exponential averaging (α=0.5) |
| Round-Robin | Baseline | Preemptive, quantum=3 ticks |
| MLFQ | Baseline | 3 levels, quanta 3/6/12, aging 20 |
| **R-MCTS** | **Proposed** | **Offline MCTS + contrastive value learning** |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{babansoro2026rmcts,
  title   = {Zero-Shot Multi-Core Process Scheduling via Offline Monte Carlo
             Tree Search with Contrastive Value Learning},
  author  = {Babansoro Sanusi, Mustapha and Embafresu, Abrha Gebriye},
  year    = {2026},
  url     = {https://doi.org/10.5281/zenodo.21317887},
  note    = {Preprint. School of Computer Science and Engineering, UESTC}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

- **Sanusi Mustapha Babansoro** — School of Computer Science and Engineering, UESTC
  (202524080114@std.uestc.edu.cn)
- **Abrha Gebriye Embafresu** — School of Computer Science and Engineering, UESTC
  (202524080110@std.uestc.edu.cn)
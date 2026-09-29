#!/usr/bin/env bash
# run.sh — generate the focused grid that lets us plot performance
# INDEPENDENTLY against number of rows and number of columns, AND across
# ensemble size, plus a unified offload-mode comparison (GPU + CPU).
#
# All runs:
#   task=classification, n_classes=10, n_jobs=4, n_repeat=3 (averaged).
#
# Two benchopt invocations (each lists datasets/solvers more than once to
# run a UNION of grids rather than a full cartesian product):
#
#   Command A — scaling sweeps (cuda, offload=gpu), kv_cache ON/OFF,
#     n_estimators ∈ {1,2,4}, n_test_samples ∈ {2,30,500,2000,8000}:
#     * rows sweep:    n_train ∈ {300,1000,2000,4000,8000,16000} at n_features=100
#     * columns sweep: n_features ∈ {20,40,100,200,500} at n_train=1000
#     → 55 datasets × 3 n_est × 2 kv_cache = 330 cells × 3 repeats = 990 runs
#
#   Command B — offload modes (GPU {gpu,cpu,disk} + CPU {cpu,disk}) at
#     n_features=200, n_test=200, n_est=4, kv_cache=False:
#     * n_train ∈ {2000,4000}  (tractable on both GPU and CPU within 16 GB RAM)
#     → 2 datasets × (3 + 2) offload = 10 cells × 3 repeats = 30 runs
#
#   Command C — GPU offload at large n_test (cuda, offload ∈ {gpu,cpu,disk}).
#     Large n_test produces large output tensors that pressure VRAM, making
#     offload modes show meaningful differences. n_features=200, n_est=4,
#     kv_cache=False:
#     * n_train ∈ {2000,4000,8000}, n_test ∈ {10000}
#     → 3 datasets × 3 offload = 9 cells × 3 repeats = 27 runs
#
# Total: 1017 runs. ~45-75 min on a single L4.
#
# Usage:
#   ./run.sh                       # uses SCRATCH_DIR default (see below)
#   SCRATCH_DIR=/scratch/tabicl ./run.sh
#
# The disk offload configs need a writable scratch dir; it is created if
# missing. Outputs land in outputs/ as benchopt_run_<timestamp>.parquet.

set -euo pipefail

# Where the benchmark repo lives (this script's directory).
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

# Shared scratch for disk offload. Override with the SCRATCH_DIR env var.
SCRATCH_DIR="${SCRATCH_DIR:-/tmp/opencode/tabicl-scratch}"
mkdir -p "$SCRATCH_DIR" outputs

# Common, fixed knobs (set on every solver selector so the runs are explicit
# rather than relying on the solver's default grid).
SOLVER_FIXED="n_jobs=4"
DATASET_FIXED="task=classification,n_classes=10"
N_REPEAT=3

echo "=============================================================="
echo " Command A — rows & columns scaling + n_estimators (cuda, gpu, kv T/F)"
echo "   n_est ∈ {1,2,4}, n_test ∈ {2, 30, 500, 2000, 8000}"
echo "   rows sweep: n_features=100, cols sweep: n_features up to 500"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=[300,1000,2000,4000,8000,16000],n_features=100,n_test_samples=[2,30,500,2000,8000],${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=1000,n_features=[20,40,100,200,500],n_test_samples=[2,30,500,2000,8000],${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=[1,2,4],kv_cache=[False,True],offload_mode=gpu,device=cuda,${SOLVER_FIXED}]" \
  -o "TabICL inference[scratch_dir=${SCRATCH_DIR},objective_label='rows & cols scaling + n_estimators, cuda, offload=gpu, kv cache on/off']" \
  -r $N_REPEAT -n 1 --no-plot

echo
echo "=============================================================="
echo " Command B — offload modes: GPU (gpu/cpu/disk) + CPU (cpu/disk)"
echo "   n_features=200, n_test=200, n_est=4, kv=False"
echo "   n_train ∈ {2000,4000}  (tractable on both GPU and CPU)"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=[2000,4000],n_features=200,n_test_samples=200,${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=4,kv_cache=False,offload_mode=['gpu','cpu','disk'],device=cuda,${SOLVER_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=4,kv_cache=False,offload_mode=['cpu','disk'],device=cpu,${SOLVER_FIXED}]" \
  -o "TabICL inference[scratch_dir=${SCRATCH_DIR},objective_label='offload modes GPU vs CPU, kv cache off, large scale']" \
  -r $N_REPEAT -n 1 --no-plot

echo
echo "=============================================================="
echo " Command C — GPU offload at large n_test (cuda, offload ∈ {gpu,cpu,disk})"
echo "   n_features=200, n_est=4, kv=False"
echo "   n_train ∈ {2000,4000,8000}, n_test ∈ {10000,20000}"
echo "   Large output tensors should pressure VRAM and make offload modes differ."
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=[2000,4000,8000],n_features=200,n_test_samples=[10000],${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=4,kv_cache=False,offload_mode=['gpu','cpu','disk'],device=cuda,${SOLVER_FIXED}]" \
  -o "TabICL inference[scratch_dir=${SCRATCH_DIR},objective_label='GPU offload at large n_test, kv cache off']" \
  -r $N_REPEAT -n 1 --no-plot

echo
echo "Done. Consolidate the new parquets with, e.g.:"
echo "  python consolidate_result_csv.py outputs/benchopt_run_*.parquet"

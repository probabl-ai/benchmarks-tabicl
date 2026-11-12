#!/usr/bin/env bash
# run.sh — compute-time case study for the TabICL gallery.
#
# Goal: give readers an order of magnitude for the walltime they can expect in
# typical conditions. The estimator runs untouched: n_estimators=8 (the
# library default), offload_mode='auto', use_amp='auto', n_jobs=None (PyTorch's
# default thread count, i.e. the physical cores), and the default auto-batching
# left ON. Only compute time is studied here; VRAM peaks are not
# (they are an artefact of the batching policy, which makes them brittle to
# interpret).
#
# Two runs. The core set produces the case-study figures; the extended set is
# the public database.
#
#   CORE (default)                       ~3 h on an L4
#   --------------
#   1. GPU scaling   rows / columns / test size, 10-class classification
#   2. GPU memory    simulated available-memory budgets, same workload
#   3. CPU scaling   rows / columns / test size, plus one GPU-matched point
#   4. small loads   rows / features / test, both devices, ~0.4-5.5 s band
#
#   EXTENDED (--full)                    ~6 h (estimate), same machine
#   -----------------
#   5. classes       n_classes ∈ {10,30,100,300,1000}, at two dataset sizes
#   6. factorial     rows × columns × test, 2×2×2 (validates the cost model)
#   7. many-class    n_classes=100, 9-point rows / columns / test
#   8. many-class    memory budgets, small workload
#   9. many-class    memory budgets, large workload
#  10. regression    9-point rows / columns / test / large-train test slices
#  11. regression    memory budgets, small workload
#  12. regression    memory budgets, large workload
#  13. estimators    n_estimators ∈ {1,2,4,8}, classification
#  14. estimators    n_estimators ∈ {1,2,4,8}, regression
#
# Design notes
# ------------
# * Every core scaling slice varies ONE dimension over five log-spaced points.
#   Every extended slice uses NINE points: the five core abscissae plus four
#   geometric midpoints, so the dense curves contain the core points exactly
#   and the two sets can be plotted together.
# * The extended set also adds a second anchor for regression (test size at
#   n_train=24000), a second size for
#   the class sweep, a 2×2×2 factorial (the one design that can validate the
#   bilinear cost model the tutorial extrapolates with), and one new axis (the
#   ensemble size) that the core study fixes at the default.
# * The column stage materialises its output buffer at rows × features ×
#   n_estimators before splitting it, so at 8 estimators the wide-and-large
#   corners are memory-bound, not time-bound. The columns slices therefore use
#   1000 test rows (2000 total) and the factorial uses 8000 rows × 500 features.
#   Anything larger would need disk offload, which is not the default and would
#   change the walltime being measured. For the same reason there is no wide
#   many-class rows anchor: 4900 rows × 400 features already peaks at 18.6 GB.
# * The fourth core slice covers the fast regime, where the study otherwise
#   has no data. It mirrors the large GPU scaling: rows, features and test are
#   each swept over five points while the other two stay at a moderate value
#   (100 rows, 10 features, 100 test rows), and every dataset runs on CPU and
#   GPU so the two devices are compared on identical workloads. The CPU floor
#   is ~0.7 s at 8 estimators (model load and per-estimator overhead), so
#   0.1 s is not reachable on CPU; the grids span the band that is. The cache
#   is on there, which moves the work into fit and leaves predict short, so
#   the fit/predict split is the informative part; one mode keeps it readable.
# * Memory sweeps run at two workload sizes, because a low budget is the only
#   place where the auto-batcher's cost can show up: the small workload shows
#   that splitting is nearly free, the larger one probes where it starts to
#   bite.
#
# Usage
# -----
#   ./run.sh                     # core study only (~3 h)
#   ./run.sh --full              # core + extended database (~6 h)
#
# There are no other flags and no environment overrides: the grids below are
# the grids that produce the published results.
#
# Cache modes and constraints
# ---------------------------
# The core and the regression scaling run the three cache modes
# (kv_cache=False / 'repr' / True); the small-workload slice runs with the
# cache on only, so the CPU and GPU curves are not multiplied by the modes.
# The many-class slices run with the cache
# off: it supports at most 10 classes and the solver skips the other modes.
# Regression pins n_classes=10 (the datum is only meaningful for
# classification, and other values would just duplicate the instances).
#
# The memory slices always run with kv_cache=False. A cache sets the memory
# floor by itself (~0.11 MB per training row, so ~0.9 GB at 8000 rows) and is
# built inside the simulated budget; without it the floor is the model plus one
# activation, i.e. a few hundred MB, and the split can get arbitrarily fine.
#
# `available_memory` (MB) simulates a smaller card by occupying VRAM until only
# that much is free, which is what makes the auto-batcher split the work (see
# benchmark_utils/base_solver.py). `None` is the real device, i.e. the
# reference point.
#
# Memory floors, measured on the L4 at n_estimators=8, kv_cache=False:
#
#   core (10 classes, 8000×200×1000, 1000 test rows)
#     390 MB  OOM (hard wall)
#     400 MB  runs: 693 row batches (~11.5 rows each)
#     450 MB  runs: 429 row batches (~19 rows each)      <- used, 60 MB margin
#
#   extended small (8000×200×1000, 1000 test rows)
#     regression 500 MB OOM, 550 MB runs (273 row batches)
#     100 classes 500 MB runs (900 row batches)         <- 550 MB used for both
#
#   extended large (16000×200×1000, 1000 test rows)
#     100 classes: 700 MB OOM, 800 MB runs (436 row batches)
#     regression:  800 MB OOM, 900 MB runs (166 row batches)  <- 1000 MB used
#
# The floor is a step: below the ICL table's one-batch cost, no split can save
# the run. The chosen values leave 50-100 MB above the measured wall, which
# matters because the real run repeats each cell three times.
#
# Caveat: every simulated budget must stay below the free VRAM measured at the
# start of each run, otherwise the cell errors out. On a clean 24 GB L4 that is
# ~22.3 GB, so the largest simulated value here (12000 MB) leaves a wide margin.
#
# Outputs
# -------
# One parquet per invocation, in outputs/ (benchopt appends _1, _2 on re-runs):
#   compute_time_{gpu_scaling,gpu_memory,cpu_scaling,
#                 small_workloads}.parquet                           (core)
#   extra_gpu_{classes,factorial,manyclass_scaling,manyclass_memory,
#              manyclass_memory_large,regression,regression_memory,
#              regression_memory_large,estimators,
#              estimators_regression}.parquet                       (--full)
# Consolidate with:
#   python consolidate_result_csv.py outputs/compute_time_*.parquet \
#     > results/results.csv                     # core only
#   python consolidate_result_csv.py outputs/compute_time_*.parquet \
#     outputs/extra_gpu_*.parquet > results/results.csv            # --full

set -euo pipefail

# --- arguments --------------------------------------------------------------
FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    -h|--help)
      awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--full]" >&2
      exit 2
      ;;
  esac
done

# Where the benchmark repo lives (this script's directory).
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

mkdir -p outputs

# --- fixed settings ---------------------------------------------------------
# Library defaults, held fixed everywhere: 8 estimators, auto offload, auto AMP.
# The ensemble size is the one axis varied in sections 12-13.
N_EST="[8]"
KV_MODES="[False,repr,True]"
KV_MODES_NOCACHE="[False,True]"
N_REPEAT=3
DATASET_FIXED="task=classification,n_classes=10"

# --- core: GPU slices -------------------------------------------------------
# Five log-spaced points per axis.
# rows:     n_features=100, n_test=2000
# columns:  n_train=1000,   n_test=2000
# test:     n_train=4000,   n_features=100
GPU_ROWS_TRAIN="[1000,3000,8000,24000,64000]"
GPU_COLS_FEATURES="[100,250,600,1600,4000]"
GPU_TEST_SAMPLES="[200,700,2500,9000,32000]"

# --- core: GPU memory sweep -------------------------------------------------
# Train-heavy on purpose (8000 training rows, 1000 test rows), kv_cache=False
# only, so nothing but the model and one activation has to fit.
#
# Measured on the L4 at n_estimators=8 (predict batches, columns/rows):
#   2000 -> 408 / 196    900 -> 408 / 237    450 -> 408 / 429
# The ICL stage always splits by ensemble member (4 batches at 8 estimators).
AV_MEM_SWEEP="[None,12000,5000,2000,900,450]"

# The memory dataset is shared by the core sweep and the extended ones.
MEM_N_TRAIN=8000
MEM_N_FEATURES=200
MEM_N_TEST=1000

# --- core: CPU slices -------------------------------------------------------
# Sized from the measured CPU cost (~0.12 ms per row x feature at the default
# thread count).
# The fourth dataset is the GPU columns anchor (1000 rows, 100 features, 1000
# test rows), so the two devices can be compared on one identical workload.
CPU_ROWS_TRAIN="[100,250,600,1600,4000]"
CPU_COLS_FEATURES="[10,25,55,130,300]"
CPU_TEST_SAMPLES="[50,150,450,1350,4000]"

# --- core: small workloads --------------------------------------------------
# Three one-axis slices at a moderate shape (100 rows, 10 features, 100 test
# rows), mirroring the large GPU scaling. Features start at 4: the 10-class
# generator needs 2**n_features >= n_classes. Measured on the L4 (3 reps):
# CPU 0.82 -> 5.45 s, GPU 0.41 -> 0.94 s; the CPU floor is ~0.7 s at 8
# estimators, so this is the reachable fast band.
SMALL_ROWS_TRAIN="[10,30,100,300,1000]"
SMALL_FEATURES="[4,8,16,32,64]"
SMALL_TEST_SAMPLES="[10,30,100,300,1000]"
SMALL_N_TRAIN=100
SMALL_N_FEATURES=10
SMALL_N_TEST=100

# --- extended: class counts -------------------------------------------------
# Two training sizes. 5000 rows = 5 rows per class at 1000 classes, the minimum
# that lets every class appear in the training split (the hierarchical fit
# needs that); 20000 rows gives a second point on the classes x rows surface.
CLASS_SWEEP="[10,30,100,300,1000]"
CLS_TRAIN_SIZES="[5000,20000]"
CLS_N_FEATURES=100
CLS_N_TEST=200

# --- extended: 2x2x2 factorial ----------------------------------------------
# One corner per combination of a low/high row count, column count and test
# size. The one-axis slices cannot distinguish a bilinear cost model from an
# additive one; these eight cells can.
FACT_ROWS="[1000,8000]"
FACT_FEATURES="[100,500]"
FACT_TEST="[200,4000]"

# --- extended: nine-point grids (core points + geometric midpoints) ---------
# rows:    1000 · 1700 · 3000 · 4900 · 8000 · 13900 · 24000 · 39000 · 64000
# columns: 100 · 160 · 250 · 390 · 600 · 980 · 1600 · 2500 · 4000
# test:    200 · 370 · 700 · 1300 · 2500 · 4700 · 9000 · 17000 · 32000
MC_ROWS_TRAIN="[1000,1700,3000,4900,8000,13900,24000,39000,64000]"
MC_COLS_FEATURES="[100,160,250,390,600,980,1600,2500,4000]"
MC_TEST_SAMPLES="[200,370,700,1300,2500,4700,9000,17000,32000]"

# --- extended: regression grids ---------------------------------------------
# Same rows and columns grids; the test axis stops at 16000 because the
# 999-quantile output is ~100x wider than 10 class logits.
REG_ROWS_TRAIN="[1000,1700,3000,4900,8000,13900,24000,39000,64000]"
REG_COLS_FEATURES="[100,160,250,390,600,980,1600,2500,4000]"
REG_TEST_SAMPLES="[200,350,600,1000,1800,3100,5400,9300,16000]"

# Second anchor for the regression regime: test-size slice at a large train set.
REG_LARGE_TRAIN=24000

# --- extended: memory sweeps at two workload sizes --------------------------
# kv_cache=False again (a cache sets the floor, and above 10 classes it is not
# supported at all). Two sizes, because splitting is nearly free on the small
# one and only starts to cost on the larger one:
#   small: 8000 train  -> floor 550 MB (regression OOMs at 500)
#   large: 16000 train -> floor 1000 MB (regression OOMs at 800)
MEM_BUDGETS="[None,12000,5000,2000,900,550]"
MEM_BUDGETS_LARGE="[None,12000,5000,2000,1500,1000]"
MEM_LARGE_N_TRAIN=16000

# --- extended: ensemble-size axis -------------------------------------------
# The core study fixes n_estimators=8; here it becomes the varied dimension,
# at two dataset sizes and for both tasks. Cache off: the cache modes are
# already covered by the core and regression slices.
N_EST_SWEEP="[1,2,4,8]"
EST_TRAIN_SIZES="[4000,24000]"
EST_N_FEATURES=100

# ============================================================================
# CORE
# ============================================================================

echo "=============================================================="
echo " 1. GPU scaling — rows / columns / test-size, auto-batching on"
echo "    rows:    n_train=${GPU_ROWS_TRAIN}, n_features=100, n_test=2000"
echo "    columns: n_features=${GPU_COLS_FEATURES}, n_train=1000, n_test=1000"
echo "    test:    n_test=${GPU_TEST_SAMPLES}, n_train=4000, n_features=100"
echo "    kv_cache=${KV_MODES}, n_estimators=${N_EST}, reps=${N_REPEAT}"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=${GPU_ROWS_TRAIN},n_features=100,n_test_samples=2000,${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=1000,n_features=${GPU_COLS_FEATURES},n_test_samples=1000,${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=4000,n_features=100,n_test_samples=${GPU_TEST_SAMPLES},${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=${KV_MODES},offload_mode=auto,device=cuda,available_memory=None]" \
  -o "TabICL inference[objective_label='compute time: rows/cols/test scaling, cuda, auto-batching, kv off/repr/on']" \
  -r "${N_REPEAT}" --no-plot \
  --output compute_time_gpu_scaling

echo
echo "=============================================================="
echo " 2. GPU memory — available_memory ∈ ${AV_MEM_SWEEP} MB"
echo "    n_train=${MEM_N_TRAIN}, n_features=${MEM_N_FEATURES}, n_test=${MEM_N_TEST}"
echo "    kv_cache=False (the cache would set the floor), n_estimators=${N_EST}, reps=${N_REPEAT}"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=${MEM_N_TRAIN},n_features=${MEM_N_FEATURES},n_test_samples=${MEM_N_TEST},${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda,available_memory=${AV_MEM_SWEEP}]" \
  -o "TabICL inference[objective_label='compute time: available memory sweep, cuda, auto-batching, kv cache off']" \
  -r "${N_REPEAT}" --no-plot \
  --output compute_time_gpu_memory

echo
echo "=============================================================="
echo " 3. CPU scaling — rows / columns / test-size, auto-batching N/A"
echo "    rows:    n_train=${CPU_ROWS_TRAIN}, n_features=100, n_test=200"
echo "    columns: n_features=${CPU_COLS_FEATURES}, n_train=1000, n_test=200"
echo "    test:    n_test=${CPU_TEST_SAMPLES}, n_train=500, n_features=100"
echo "    matched: n_train=1000, n_features=100, n_test=1000 (GPU columns anchor)"
echo "    kv_cache=${KV_MODES}, n_estimators=${N_EST}, reps=${N_REPEAT}"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=${CPU_ROWS_TRAIN},n_features=100,n_test_samples=200,${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=1000,n_features=${CPU_COLS_FEATURES},n_test_samples=200,${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=500,n_features=100,n_test_samples=${CPU_TEST_SAMPLES},${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=1000,n_features=100,n_test_samples=1000,${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=${KV_MODES},offload_mode=auto,device=cpu]" \
  -o "TabICL inference[objective_label='compute time: rows/cols/test scaling, cpu, kv off/repr/on']" \
  -r "${N_REPEAT}" --no-plot \
  --output compute_time_cpu_scaling

echo
echo "=============================================================="
echo " 4. Small workloads — rows / features / test, CPU and GPU, fast regime"
echo "    rows:     n_train=${SMALL_ROWS_TRAIN}, n_features=${SMALL_N_FEATURES}, n_test=${SMALL_N_TEST}"
echo "    features: n_features=${SMALL_FEATURES}, n_train=${SMALL_N_TRAIN}, n_test=${SMALL_N_TEST}"
echo "    test:     n_test=${SMALL_TEST_SAMPLES}, n_train=${SMALL_N_TRAIN}, n_features=${SMALL_N_FEATURES}"
echo "    same datasets on both devices; kv_cache=True, n_estimators=${N_EST}, reps=${N_REPEAT}"
echo "=============================================================="
benchopt run . \
  -d "Simulated[n_train_samples=${SMALL_ROWS_TRAIN},n_features=${SMALL_N_FEATURES},n_test_samples=${SMALL_N_TEST},${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=${SMALL_N_TRAIN},n_features=${SMALL_FEATURES},n_test_samples=${SMALL_N_TEST},${DATASET_FIXED}]" \
  -d "Simulated[n_train_samples=${SMALL_N_TRAIN},n_features=${SMALL_N_FEATURES},n_test_samples=${SMALL_TEST_SAMPLES},${DATASET_FIXED}]" \
  -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=True,offload_mode=auto,device=[cpu,cuda],available_memory=None]" \
  -o "TabICL inference[objective_label='compute time: small workloads, rows/features/test, cpu+cuda, kv cache on']" \
  -r "${N_REPEAT}" --no-plot \
  --output compute_time_small_workloads

# ============================================================================
# EXTENDED (--full)
# ============================================================================

if [ "$FULL" = "1" ]; then
  echo
  echo "=============================================================="
  echo " 5. class count — n_classes ∈ ${CLASS_SWEEP}, n_train ∈ ${CLS_TRAIN_SIZES}"
  echo "    n_features=${CLS_N_FEATURES}, n_test=${CLS_N_TEST}"
  echo "    kv_cache=False (the cache supports <= 10 classes), reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${CLS_TRAIN_SIZES},n_features=${CLS_N_FEATURES},n_test_samples=${CLS_N_TEST},n_classes=${CLASS_SWEEP},task=classification]" \
    -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: class-count sweep, two dataset sizes, cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_classes

  echo
  echo "=============================================================="
  echo " 6. factorial — rows ${FACT_ROWS} × columns ${FACT_FEATURES} × test ${FACT_TEST}"
  echo "    10-class classification, kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${FACT_ROWS},n_features=${FACT_FEATURES},n_test_samples=${FACT_TEST},${DATASET_FIXED}]" \
    -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: 2x2x2 rows x columns x test factorial, cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_factorial

  echo
  echo "=============================================================="
  echo " 7. many-class scaling — n_classes=100, nine-point slices"
  echo "    rows:       n_train=${MC_ROWS_TRAIN}, n_features=100, n_test=2000"
  echo "    columns:    n_features=${MC_COLS_FEATURES}, n_train=1000, n_test=1000"
  echo "    test:       n_test=${MC_TEST_SAMPLES}, n_train=4000, n_features=100"
  echo "    kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${MC_ROWS_TRAIN},n_features=100,n_test_samples=2000,n_classes=100,task=classification]" \
    -d "Simulated[n_train_samples=1000,n_features=${MC_COLS_FEATURES},n_test_samples=1000,n_classes=100,task=classification]" \
    -d "Simulated[n_train_samples=4000,n_features=100,n_test_samples=${MC_TEST_SAMPLES},n_classes=100,task=classification]" \
    -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: many-class scaling (100 classes), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_manyclass_scaling

  echo
  echo "=============================================================="
  echo " 8. many-class memory — n_classes=100, small workload"
  echo "    n_train=${MEM_N_TRAIN}, n_features=${MEM_N_FEATURES}, n_test=${MEM_N_TEST}"
  echo "    available_memory ∈ ${MEM_BUDGETS} MB, kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${MEM_N_TRAIN},n_features=${MEM_N_FEATURES},n_test_samples=${MEM_N_TEST},n_classes=100,task=classification]" \
    -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda,available_memory=${MEM_BUDGETS}]" \
    -o "TabICL inference[objective_label='compute time: available memory sweep (100 classes), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_manyclass_memory

  echo
  echo "=============================================================="
  echo " 9. many-class memory — n_classes=100, large workload"
  echo "    n_train=${MEM_LARGE_N_TRAIN}, n_features=${MEM_N_FEATURES}, n_test=${MEM_N_TEST}"
  echo "    available_memory ∈ ${MEM_BUDGETS_LARGE} MB, kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${MEM_LARGE_N_TRAIN},n_features=${MEM_N_FEATURES},n_test_samples=${MEM_N_TEST},n_classes=100,task=classification]" \
    -s "TabICL-Classifier[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda,available_memory=${MEM_BUDGETS_LARGE}]" \
    -o "TabICL inference[objective_label='compute time: available memory sweep, large workload (100 classes), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_manyclass_memory_large

  echo
  echo "=============================================================="
  echo " 10. regression scaling — nine-point slices"
  echo "    rows:       n_train=${REG_ROWS_TRAIN}, n_features=100, n_test=1000"
  echo "    columns:    n_features=${REG_COLS_FEATURES}, n_train=1000, n_test=1000"
  echo "    test:       n_test=${REG_TEST_SAMPLES}, n_train=4000, n_features=100"
  echo "    large train test: n_test=${REG_TEST_SAMPLES}, n_train=${REG_LARGE_TRAIN}, n_features=100"
  echo "    kv_cache=${KV_MODES_NOCACHE}, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${REG_ROWS_TRAIN},n_features=100,n_test_samples=1000,n_classes=10,task=regression]" \
    -d "Simulated[n_train_samples=1000,n_features=${REG_COLS_FEATURES},n_test_samples=1000,n_classes=10,task=regression]" \
    -d "Simulated[n_train_samples=4000,n_features=100,n_test_samples=${REG_TEST_SAMPLES},n_classes=10,task=regression]" \
    -d "Simulated[n_train_samples=${REG_LARGE_TRAIN},n_features=100,n_test_samples=${REG_TEST_SAMPLES},n_classes=10,task=regression]" \
    -s "TabICL-Regressor[n_estimators=${N_EST},kv_cache=${KV_MODES_NOCACHE},offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: regression scaling, cuda, kv off/on']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_regression

  echo
  echo "=============================================================="
  echo " 11. regression memory — small workload"
  echo "    n_train=${MEM_N_TRAIN}, n_features=${MEM_N_FEATURES}, n_test=${MEM_N_TEST}"
  echo "    available_memory ∈ ${MEM_BUDGETS} MB, kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${MEM_N_TRAIN},n_features=${MEM_N_FEATURES},n_test_samples=${MEM_N_TEST},n_classes=10,task=regression]" \
    -s "TabICL-Regressor[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda,available_memory=${MEM_BUDGETS}]" \
    -o "TabICL inference[objective_label='compute time: available memory sweep (regression), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_regression_memory

  echo
  echo "=============================================================="
  echo " 12. regression memory — large workload"
  echo "    n_train=${MEM_LARGE_N_TRAIN}, n_features=${MEM_N_FEATURES}, n_test=${MEM_N_TEST}"
  echo "    available_memory ∈ ${MEM_BUDGETS_LARGE} MB, kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${MEM_LARGE_N_TRAIN},n_features=${MEM_N_FEATURES},n_test_samples=${MEM_N_TEST},n_classes=10,task=regression]" \
    -s "TabICL-Regressor[n_estimators=${N_EST},kv_cache=False,offload_mode=auto,device=cuda,available_memory=${MEM_BUDGETS_LARGE}]" \
    -o "TabICL inference[objective_label='compute time: available memory sweep, large workload (regression), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_regression_memory_large

  echo
  echo "=============================================================="
  echo " 13. ensemble size — classification, n_estimators ∈ ${N_EST_SWEEP}"
  echo "    n_train=${EST_TRAIN_SIZES}, n_features=${EST_N_FEATURES}, n_test=2000"
  echo "    kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${EST_TRAIN_SIZES},n_features=${EST_N_FEATURES},n_test_samples=2000,${DATASET_FIXED}]" \
    -s "TabICL-Classifier[n_estimators=${N_EST_SWEEP},kv_cache=False,offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: ensemble-size sweep, cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_estimators

  echo
  echo "=============================================================="
  echo " 14. ensemble size — regression, n_estimators ∈ ${N_EST_SWEEP}"
  echo "    n_train=${EST_TRAIN_SIZES}, n_features=${EST_N_FEATURES}, n_test=1000"
  echo "    kv_cache=False, reps=${N_REPEAT}"
  echo "=============================================================="
  benchopt run . \
    -d "Simulated[n_train_samples=${EST_TRAIN_SIZES},n_features=${EST_N_FEATURES},n_test_samples=1000,n_classes=10,task=regression]" \
    -s "TabICL-Regressor[n_estimators=${N_EST_SWEEP},kv_cache=False,offload_mode=auto,device=cuda]" \
    -o "TabICL inference[objective_label='compute time: ensemble-size sweep (regression), cuda, kv cache off']" \
    -r "${N_REPEAT}" --no-plot \
    --output extra_gpu_estimators_regression
fi

echo
if [ "$FULL" = "1" ]; then
  echo "Done (core + extended). Consolidate with:"
  echo "  python consolidate_result_csv.py outputs/compute_time_*.parquet \\"
  echo "    outputs/extra_gpu_*.parquet > results/results.csv"
else
  echo "Done (core). Consolidate with:"
  echo "  python consolidate_result_csv.py outputs/compute_time_*.parquet \\"
  echo "    > results/results.csv"
  echo
  echo "Run './run.sh --full' for the many-class, regression and ensemble sets."
fi

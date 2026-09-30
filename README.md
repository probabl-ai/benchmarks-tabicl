# benchmarks-tabicl

Benchopt benchmark measuring **execution walltime and memory footprint** of
[TabICL](https://tabicl.readthedocs.io) classifiers and regressors across
**dataset sizes and hardware**.

It records, per `(dataset, solver, parameters, repetition)` cell:

| What | Where it comes from |
| --- | --- |
| Solver walltime (`time`) | benchopt, built-in |
| `fit_time`, `predict_time` | this benchmark (split inside `Solver.run`) |
| Peak **RAM** (`ram_peak_mb`) | this benchmark (psutil sampling) |
| Peak **VRAM** (`vram_peak_mb`) | this benchmark (`torch.cuda.max_memory_allocated`) |
| Accuracy / log-loss, RMSE / R2 | this benchmark (sklearn) |
| Run label (`p_objective_objective_label`) | benchopt, from the required `objective_label` parameter |
| `torch` / `tabicl` versions | this benchmark (`objective_torch_version`, `objective_tabicl_version`) |
| CPU model, core count, system RAM, CUDA version, run date | benchopt provenance (automatic) |
| GPU name + selected device | this benchmark (`objective_gpu_name`, `objective_device`) |

> **`objective_label` is required.** Every run must describe *why* it is
> performed via the `objective_label` objective parameter, e.g.
> `-o "TabICL inference[objective_label='baseline CPU vs GPU']"`. Leaving it
> unset raises a `ValueError`. The label is recorded as
> `p_objective_objective_label` in the output parquet.

> TabICL does not *train*: `fit` stores/prepares data, learning happens in
> `predict` via in-context learning. Both are timed inside `run`; `fit_time`
> and `predict_time` are reported separately on top of benchopt's `time`.

## Structure

```
objective.py              # what is scored: timings, RAM/VRAM, accuracy/RMSE
datasets/simulated.py      # synthetic classification/regression grids
solvers/tabicl_classifier.py
solvers/tabicl_regressor.py
benchmark_utils/           # RAM/VRAM tracker, metrics, GPU info (shipped)
run.sh                     # focused gallery grid (scaling + offload comparisons)
config_run.yml             # example run configuration
test_config.py
display/gallery_figures.py # Plotly figures for the sphinx gallery
consolidate_result_csv.py  # parquet → CSV consolidation + Google Sheets sync
```

## Install

```bash
# Into the current env (needs torch; on a GPU machine use a CUDA torch build):
benchopt install .

# Or into an isolated conda env pinned to python 3.12 (recommended on shared
# GPU boxes):
benchopt install . -e
benchopt run . -e
```

Download the TabICL checkpoint once (it is cached afterwards):

```bash
benchopt prepare .          # runs Dataset.prepare (no-op for Simulated)
```

## Run

```bash
# Smoke test on the tiny test config:
benchopt run . -d Simulated -s TabICL-Classifier -n 1

# Full grid (both tasks, the default size ladder, 3 repetitions):
benchopt run . --config config_run.yml

# Restrict to one size and force CPU to compare hardware on the same machine:
benchopt run . \
    -d "Simulated[n_samples=5000,n_features=50,task=classification]" \
    -s "TabICL-Classifier[device=cpu]"
```

### Comparing hardware

Each run is self-contained: the result parquet carries its own provenance
(CPU model, cores, RAM, CUDA version, GPU name). To aggregate runs from
several machines:

```bash
# Collect all parquets in outputs/, then:
benchopt merge --keep all      # keep every machine's rows (compare hardware)
benchopt plot                  # regenerate the dashboard
```

Use `--keep all` (not the default `last`) so identical configs run on
different machines are not collapsed.

### `run.sh` — the focused gallery grid

`run.sh` is a convenience script that runs the three-command grid used to
produce the gallery figures (`display/gallery_figures.py`). All runs are
classification-only, `n_classes=10`, `n_jobs=4`, `n_repeat=3` (averaged).

```bash
./run.sh                       # uses the default scratch dir
SCRATCH_DIR=/scratch/tabicl ./run.sh   # override the disk-offload scratch dir
```

| Command | What it sweeps | Cells (×3 reps) |
| --- | --- | --- |
| **A — scaling** | `n_train ∈ {300,1k,2k,4k,8k,16k}` at `n_features=100`; `n_features ∈ {20,40,100,200,500}` at `n_train=1000`; `n_est ∈ {1,2,4}`, `kv_cache T/F`, `n_test ∈ {2,30,500,2k,8k}` | 990 |
| **B — offload (GPU+CPU)** | `n_train ∈ {2000,4000}`, `n_features=200`, `n_test=200`, `n_est=4`, `kv=False`; GPU `offload ∈ {gpu,cpu,disk}` + CPU `offload ∈ {cpu,disk}` | 30 |
| **C — offload at large n_test (GPU)** | `n_train ∈ {2000,4000,8000}`, `n_features=200`, `n_test=10000`, `n_est=4`, `kv=False`; GPU `offload ∈ {gpu,cpu,disk}` | 27 |

**Total: 1047 runs, ~45-75 min on a single L4.** Outputs land in `outputs/`
as `benchopt_run_<timestamp>.parquet` (one per command). Consolidate with:

```bash
python consolidate_result_csv.py outputs/benchopt_run_*.parquet \
    --sync-to-gspread \
    --gspread-url "https://docs.google.com/spreadsheets/d/..." \
    --gspread-auth-key .gspread-service-account-key/tabicl-benchmarks-*.json
```

Notes:
- Command B keeps `n_train ≤ 4000` so CPU runs stay within a 16 GB host RAM
  budget (CPU `offload=disk` is a no-op — the InferenceManager bypasses
  offload logic for `device='cpu'`).
- Command C is GPU-only at large `n_test` so the output tensor pressures
  VRAM and the offload modes show measurable VRAM savings.

### `display/gallery_figures.py` — gallery figures

Builds the three Plotly figures (KV cache timing, KV cache VRAM, offload
comparison) from a consolidated `results.csv`:

```bash
python display/gallery_figures.py --csv results/results.csv \
    --out display/gallery_figures
# -> display/gallery_figures/fig{1,2,3}_*.html
```

The figures are embedded into the TabICL sphinx-gallery page; see the
script's `--help` for the comparison (`--compare-csv`) and gallery-embed
(`--gallery-html`) modes.

## Results

Results land in `outputs/` as `benchopt_run_<timestamp>.parquet` plus an HTML
dashboard. Inspect in Python:

```python
from benchopt.results import read_results

df = read_results("outputs/benchopt_run_*.parquet")
# Final point per (dataset, solver, repetition):
final = (
    df.sort_values("stop_val")
    .groupby(["dataset_name", "solver_name", "idx_rep"], as_index=False)
    .last()
)
```

## Tested solvers and datasets

- **Solvers**: `TabICL-Classifier`, `TabICL-Regressor` (TabICL-only for now;
  reference baselines such as gradient-boosted trees can be added later).

  Parameter grid (each combination is a distinct result row):

  | Parameter | Values | Notes |
  | --- | --- | --- |
  | `n_estimators` | `1, 2, 4, 8` | ensemble size |
  | `batch_size` | `8` | default; extend to sweep |
  | `kv_cache` | `False, True` | cache built during `fit` |
  | `offload_mode` | `auto, gpu, cpu, disk` | `disk` uses the objective's `scratch_dir` |
  | `n_jobs` | `-1` | use all CPU cores |
  | `device` | `cpu, cuda, mps, xpu` | explicit; runs for unavailable devices are skipped |

  **Objective parameters** (set once for all solvers via `-o`):

  | Parameter | Values | Notes |
  | --- | --- | --- |
  | `objective_label` | required | short free-text label describing why this run is performed |
  | `scratch_dir` | `None` | shared scratch location; solvers that use `offload_mode='disk'` create a `disk-offload/` subdir inside it (with a per-run temp dir). Required when any solver uses `offload_mode='disk'` (skipped otherwise). |

  Each run does `fit` then `predict` inline (benchopt's `time` covers both).
  Peak RAM and peak VRAM are recorded **separately for the fit and predict
  phases** (a `ResourceTracker` is started/stopped around each), in addition
  to a combined `ram_peak_mb` / `vram_peak_mb` (max of the two windows) as the
  headline aggregate peak:

  | kv_cache | benchopt `time` | fit-window peak | predict-window peak |
  |----------|----------------|----------------|---------------------|
  | True     | fit + predict  | fit(cache)     | predict             |
  | False    | fit + predict  | fit            | predict             |

  ``fit_time`` and ``predict_time`` are always reported separately, so the
  breakdown is recoverable. The per-phase peaks (`ram_peak_fit_mb`,
  `vram_peak_predict_mb`, `vram_peak_fit_mb`, `vram_peak_predict_mb`) let the
  fit and predict memory costs be plotted independently.

  Full grid = 4 × 2 × 4 = 32 configs per dataset; restrict with `-s` for
  faster iteration. For disk offload, pass `scratch_dir=/path/to/dir` via the
  objective (e.g. `-o "TabICL inference[scratch_dir=/scratch]"`).

- **Datasets**: `Simulated` with a default grid
  `(1000, 20)`, `(5000, 50)`, `(10000, 100)`, `(50000, 100)` ×
  `{classification (n_classes ∈ {10, 100}), regression}`.
  Override per run via `-d` or `config_run.yml`.

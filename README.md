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
config_run.yml             # example run configuration
test_config.py
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
  | `warmup` | `True, False` | pre-fit + dummy predict ahead of the timed run |
  | `offload_mode` | `auto, gpu, cpu, disk` | `disk` uses the objective's `scratch_dir` |
  | `n_jobs` | `-1` | use all CPU cores |
  | `device` | `cpu, cuda, mps, xpu` | explicit; runs for unavailable devices are skipped |

  **Objective parameters** (set once for all solvers via `-o`):

  | Parameter | Values | Notes |
  | --- | --- | --- |
  | `objective_label` | required | short free-text label describing why this run is performed |
  | `scratch_dir` | `None` | shared scratch location; solvers that use `offload_mode='disk'` create a `disk-offload/` subdir inside it (with a per-run temp dir). Required when any solver uses `offload_mode='disk'` (skipped otherwise). |

  ``warmup`` and ``kv_cache`` cross to form four measurement scenarios:

  | warmup | kv_cache | benchopt `time` | peak RAM/VRAM window |
  |--------|----------|----------------|----------------------|
  | True   | True     | predict only   | fit(cache) + dummy predict + predict |
  | True   | False    | predict only   | fit + dummy predict + predict |
  | False  | True     | fit + predict  | fit(cache) + predict |
  | False  | False    | fit + predict  | fit + predict |

  ``fit_time`` and ``predict_time`` are always reported separately, so the
  breakdown is recoverable regardless of scenario. Note that benchopt's
  ``time`` column covers predict only when ``warmup=True`` and fit+predict
  when ``warmup=False`` — compare across scenarios via ``fit_time`` /
  ``predict_time``, not ``time`` alone.

  Full grid = 4 × 2 × 2 × 4 = 64 configs per dataset; restrict with `-s` for
  faster iteration. For disk offload, pass `scratch_dir=/path/to/dir` via the
  objective (e.g. `-o "TabICL inference[scratch_dir=/scratch]"`).

- **Datasets**: `Simulated` with a default grid
  `(1000, 20)`, `(5000, 50)`, `(10000, 100)`, `(50000, 100)` ×
  `{classification (n_classes ∈ {10, 100}), regression}`.
  Override per run via `-d` or `config_run.yml`.

"""Consolidate TabICL benchopt results into a CSV and sync to Google Sheets.

Reads one or more benchopt ``.parquet`` result files, aggregates across
repetitions (mean ± std when >1 rep), and either prints a CSV to stdout or
syncs to a Google Spreadsheet with the same formatting conventions as the
sklearn-engine-benchmarks project (benchmark-id hashes, bi-color blocks,
sorted by dataset-defining columns, frozen header + defining frame).

The left part of the sheet is a fixed frame that defines blocks of rows
that can be compared to each other because they solve the same use case
(same dataset config + seed). Everything to the right of the "Walltime"
column is the variable part (solver params, metrics, provenance).

Usage
-----
    # Print CSV to stdout
    python consolidate_result_csv.py outputs/*.parquet

    # Sync to Google Sheets
    python consolidate_result_csv.py outputs/results.parquet \
        --sync-to-gspread \
        --gspread-url "https://docs.google.com/spreadsheets/d/..." \
        --gspread-auth-key .gspread-service-account-key/tabicl-benchmarks-*.json
"""

import hashlib
from functools import partial
from io import BytesIO
from itertools import zip_longest

import numpy as np
import pandas as pd

GOOGLE_WORKSHEET_NAME = "TabICL"

DATES_FORMAT = "%Y-%m-%d"

# --- Column names (human-readable) ---

BENCHMARK_ID_NAME = "Benchmark id"
TASK = "Task"
NB_TRAIN_SAMPLES = "Nb train samples"
NB_FEATURES = "Nb features"
NB_TEST_SAMPLES = "Nb test samples"
NB_CLASSES = "Nb classes"
SEED = "Seed"
WALLTIME = "Walltime"
FIT_TIME = "Fit time"
PREDICT_TIME = "Predict time"
RAM_PEAK = "RAM peak (MB)"
VRAM_PEAK = "VRAM peak (MB)"
ACCURACY = "Accuracy"
LOG_LOSS = "Log loss"
RMSE = "RMSE"
R2 = "R2"
NB_ESTIMATORS = "Nb estimators"
BATCH_SIZE = "Batch size"
KV_CACHE = "KV cache"
WARMUP = "Warmup"
OFFLOAD_MODE = "Offload mode"
NB_JOBS = "Nb jobs"
DEVICE = "Device"
USE_AMP = "Use AMP"
GPU_NAME = "GPU name"
TORCH_VERSION = "Torch version"
TABICL_VERSION = "TabICL version"
RUN_LABEL = "Run label"
PLATFORM = "Platform"
PLATFORM_ARCHITECTURE = "Platform arch"
PLATFORM_RELEASE = "Platform release"
PLATFORM_VERSION = "Platform version"
SYSTEM_CPUS = "Nb cpus"
SYSTEM_PROCESSOR = "CPU name"
SYSTEM_RAM = "RAM (GB)"
CUDA_VERSION = "CUDA version"
NUMPY_VERSION = "Numpy version"
SCIPY_VERSION = "Scipy version"
RUN_DATE = "Run date"

# --- Parquet → display name mapping ---

PARQUET_TABLE_DISPLAY_MAPPING = {
    "p_dataset_task": TASK,
    "p_dataset_n_train_samples": NB_TRAIN_SAMPLES,
    "p_dataset_n_features": NB_FEATURES,
    "p_dataset_n_test_samples": NB_TEST_SAMPLES,
    "p_dataset_n_classes": NB_CLASSES,
    "objective_fit_time": FIT_TIME,
    "objective_predict_time": PREDICT_TIME,
    "objective_ram_peak_mb": RAM_PEAK,
    "objective_vram_peak_mb": VRAM_PEAK,
    "objective_accuracy": ACCURACY,
    "objective_log_loss": LOG_LOSS,
    "objective_rmse": RMSE,
    "objective_r2": R2,
    "p_solver_n_estimators": NB_ESTIMATORS,
    "p_solver_batch_size": BATCH_SIZE,
    "p_solver_kv_cache": KV_CACHE,
    "p_solver_warmup": WARMUP,
    "p_solver_offload_mode": OFFLOAD_MODE,
    "p_solver_n_jobs": NB_JOBS,
    "p_solver_device": DEVICE,
    "p_solver_use_amp": USE_AMP,
    "objective_gpu_name": GPU_NAME,
    "objective_torch_version": TORCH_VERSION,
    "objective_tabicl_version": TABICL_VERSION,
    "p_obj_objective_label": RUN_LABEL,
    "platform": PLATFORM,
    "platform-architecture": PLATFORM_ARCHITECTURE,
    "platform-release": PLATFORM_RELEASE,
    "platform-version": PLATFORM_VERSION,
    "system-cpus": SYSTEM_CPUS,
    "system-processor": SYSTEM_PROCESSOR,
    "system-ram (GB)": SYSTEM_RAM,
    "version-cuda": CUDA_VERSION,
    "version-numpy": NUMPY_VERSION,
    "version-scipy": SCIPY_VERSION,
    "run_date": RUN_DATE,
}

# --- Benchmark-defining columns (the fixed left frame) ---
# Rows sharing the same benchmark_id (hash of these) are directly comparable.
# NB_ESTIMATORS is part of the defining frame since different ensemble sizes
# produce different benchmark scenarios — what we're studying is timings.

BENCHMARK_DEFINING_COLUMNS = [
    TASK,
    NB_TRAIN_SAMPLES,
    NB_FEATURES,
    NB_TEST_SAMPLES,
    NB_CLASSES,
    NB_ESTIMATORS,
    # TODO?: re-add SEED here if multiple seeds are used in the future, so
    # that runs with different seeds get distinct benchmark IDs.
]
_benchmark_defining_columns_identifier = "".join(sorted(BENCHMARK_DEFINING_COLUMNS))

# Columns omitted from the output entirely.
OMITTED_PARQUET_COLUMNS = {
    "objective_name",
    "solver_name",
    "dataset_name",
    "obj_description",
    "solver_description",
    "file_objective",
    "file_solver",
    "file_dataset",
    "sampling_strategy",
    "stop_val",
    "idx_rep",
    "base_seed",  # TODO?: re-add if multiple seeds are used
    "time",  # use fit_time + predict_time instead of total walltime
    "objective_value",
    "objective_disk_offload_dir",
    "p_obj_scratch_dir",
    "benchmark-git-tag",
    "env-OMP_NUM_THREADS",
    "version-numpy-libs",
}

# Numeric metric columns that are averaged across repetitions.
# Primary metrics (timings + memory) come first; performance metrics
# (accuracy/log_loss/rmse/r2) are sanity checks and go to the rightmost cols.
PRIMARY_METRIC_COLUMNS = [
    FIT_TIME,
    PREDICT_TIME,
    RAM_PEAK,
    VRAM_PEAK,
]

SANITY_METRIC_COLUMNS = [
    ACCURACY,
    LOG_LOSS,
    RMSE,
    R2,
]

NUMERIC_METRIC_COLUMNS = PRIMARY_METRIC_COLUMNS + SANITY_METRIC_COLUMNS

# Solver parameter columns (part of the grouping key for averaging + dedup).
# NB_ESTIMATORS is in BENCHMARK_DEFINING_COLUMNS, not here.
SOLVER_PARAM_COLUMNS = [
    BATCH_SIZE,
    KV_CACHE,
    WARMUP,
    OFFLOAD_MODE,
    NB_JOBS,
    DEVICE,
    USE_AMP,
]

# Provenance columns (part of the dedup key).
PROVENANCE_COLUMNS = [
    GPU_NAME,
    PLATFORM,
    PLATFORM_ARCHITECTURE,
    PLATFORM_RELEASE,
    PLATFORM_VERSION,
    SYSTEM_PROCESSOR,
    SYSTEM_CPUS,
    SYSTEM_RAM,
    CUDA_VERSION,
    TORCH_VERSION,
    TABICL_VERSION,
    NUMPY_VERSION,
    SCIPY_VERSION,
    RUN_LABEL,
]

# --- Display order ---
# Left frame: benchmark id + defining columns (incl. nb_estimators)
# Pivot: primary metrics (fit_time, predict_time, ram, vram) — bold, frozen after
# Middle: remaining solver params + provenance
# Rightmost: sanity metrics (accuracy etc) + run date

TABLE_DISPLAY_ORDER = (
    [BENCHMARK_ID_NAME]
    + BENCHMARK_DEFINING_COLUMNS
    + PRIMARY_METRIC_COLUMNS
    + SOLVER_PARAM_COLUMNS
    + [GPU_NAME]
    + PROVENANCE_COLUMNS[1:]  # skip GPU_NAME (already listed)
    + SANITY_METRIC_COLUMNS
    + [RUN_DATE]
)

# --- Dedup key: if the same config was run multiple times, keep the most recent ---

UNIQUE_BENCHMARK_KEY = [BENCHMARK_ID_NAME] + SOLVER_PARAM_COLUMNS + PROVENANCE_COLUMNS

# --- Sort order (importance, ascending=True/False) ---

ROW_SORT_ORDER = [
    (TASK, True),  # increasing
    (NB_TRAIN_SAMPLES, False),  # decreasing
    (NB_FEATURES, False),  # decreasing
    (NB_TEST_SAMPLES, False),  # decreasing
    (NB_CLASSES, True),  # increasing
    (NB_ESTIMATORS, False),  # decreasing
    (FIT_TIME, True),
    (BATCH_SIZE, True),
    (KV_CACHE, True),
    (WARMUP, True),
    (OFFLOAD_MODE, True),
    (DEVICE, True),
    (USE_AMP, True),
    (GPU_NAME, True),
    (SYSTEM_CPUS, True),
    (SYSTEM_PROCESSOR, True),
    (SYSTEM_RAM, True),
    (PLATFORM, True),
    (PLATFORM_ARCHITECTURE, True),
    (PLATFORM_RELEASE, True),
    (RUN_LABEL, True),
    (RUN_DATE, False),
    (BENCHMARK_ID_NAME, True),
]
_row_sort_by, _row_sort_ascending = map(list, zip(*ROW_SORT_ORDER, strict=True))

IDS_LENGTH = 8


def _get_id_from_str(s):
    return hashlib.sha256(s.encode("utf8"), usedforsecurity=False).hexdigest()[
        :IDS_LENGTH
    ]


def _get_benchmark_id(row):
    return _get_id_from_str(
        "".join(str(row[c]) for c in BENCHMARK_DEFINING_COLUMNS)
        + _benchmark_defining_columns_identifier
    )


def _load_parquet(source):
    """Load a parquet file, rename columns, drop omitted ones."""
    df = pd.read_parquet(source)

    cols_to_keep = [c for c in df.columns if c not in OMITTED_PARQUET_COLUMNS]
    df = df[cols_to_keep]

    # Backfill columns added after the first runs: parquets produced before
    # use_amp was a solver parameter don't carry p_solver_use_amp. Fill with
    # "auto" (the estimator's default) so old and new runs aggregate together.
    if "p_solver_use_amp" not in df.columns:
        df["p_solver_use_amp"] = "auto"

    df = df.rename(columns=PARQUET_TABLE_DISPLAY_MAPPING, errors="raise")

    return df


def _aggregate_repetitions(df):
    """Average numeric metrics across repetitions (idx_rep).

    Groups by (benchmark-defining + solver params + provenance). If only one
    rep per group, values pass through unchanged. If multiple reps, numeric
    columns are averaged and formatted as ``mean±std`` strings.
    """
    group_cols = BENCHMARK_DEFINING_COLUMNS + SOLVER_PARAM_COLUMNS + PROVENANCE_COLUMNS

    if df.groupby(group_cols, dropna=False).ngroups == len(df):
        return df

    aggregated_rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        row = {}
        for c in group.columns:
            if c not in NUMERIC_METRIC_COLUMNS:
                row[c] = group[c].iloc[0]
            else:
                vals = group[c].dropna()
                if len(vals) == 0:
                    row[c] = np.nan
                elif len(vals) == 1:
                    row[c] = vals.iloc[0]
                else:
                    mean = vals.mean()
                    std = vals.std()
                    if std > 0:
                        row[c] = f"{mean:.4g}±{std:.4g}"
                    else:
                        row[c] = mean
        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


def _build_table(parquet_sources):
    """Load, aggregate, add benchmark ids, sort, dedup."""
    dfs = [_load_parquet(f) for f in parquet_sources]
    df = pd.concat(dfs, ignore_index=True, copy=False) if len(dfs) > 1 else dfs[0]

    df = _aggregate_repetitions(df)

    df[BENCHMARK_ID_NAME] = df.apply(_get_benchmark_id, axis=1)

    for col in TABLE_DISPLAY_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[TABLE_DISPLAY_ORDER]

    df[RUN_DATE] = pd.to_datetime(df[RUN_DATE], errors="coerce")

    df.sort_values(
        by=_row_sort_by, ascending=_row_sort_ascending, inplace=True, kind="stable"
    )

    df.drop_duplicates(subset=UNIQUE_BENCHMARK_KEY, inplace=True, ignore_index=True)

    df = _sanitize_df_with_tocsv(df)

    return df


def _sanitize_df_with_tocsv(df):
    """Round-trip through CSV to normalize None/NaN/empty-string values."""
    buf = BytesIO()
    _df_to_csv(df, buf)
    buf.seek(0)
    df = pd.read_csv(buf, keep_default_na=False, na_values=[""])
    df = df[TABLE_DISPLAY_ORDER]
    return df


def _df_to_csv(df, target):
    """Write df to CSV with formatted floats."""
    float_format_fn = partial(
        np.format_float_positional,
        precision=4,
        unique=True,
        fractional=False,
        trim="-",
        sign=False,
        pad_left=None,
        pad_right=None,
        min_digits=None,
    )
    df = df.copy()
    for col in NUMERIC_METRIC_COLUMNS:
        if col in df.columns and df[col].dtype.kind in "f":
            df[col] = df[col].map(float_format_fn)
    df.to_csv(target, index=False, mode="a", date_format=DATES_FORMAT)


def _gspread_sync(df, gspread_url, gspread_auth_key):
    """Sync the dataframe to a Google Spreadsheet with formatting."""
    import gspread

    n_rows, n_cols = df.shape
    pivot_col = df.columns.get_loc(VRAM_PEAK) + 1

    gs = gspread.service_account(gspread_auth_key)
    sheet = gs.open_by_url(gspread_url)

    global_range = (
        f"{gspread.utils.rowcol_to_a1(1, 1)}:"
        f"{gspread.utils.rowcol_to_a1(n_rows + 1, n_cols)}"
    )

    try:
        worksheet = sheet.worksheet(GOOGLE_WORKSHEET_NAME)
        worksheet.clear()
        worksheet.clear_basic_filter()
        worksheet.freeze(0, 0)
        worksheet.resize(rows=n_rows + 1, cols=n_cols)
        worksheet.clear_notes(global_range)
        reset_format = dict(
            backgroundColorStyle=dict(rgbColor=dict(red=1, green=1, blue=1, alpha=1)),
            textFormat=dict(bold=False),
        )
        worksheet.format(global_range, reset_format)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(
            GOOGLE_WORKSHEET_NAME, rows=n_rows + 1, cols=n_cols
        )

    # Upload values
    worksheet.update(
        values=[df.columns.values.tolist()] + df.values.tolist(), range_name="A1"
    )

    # Set filter
    worksheet.set_basic_filter(1, 1, n_rows + 1, n_cols)

    # Freeze: header row + defining columns (up to the first primary metric)
    worksheet.freeze(rows=1, cols=pivot_col)

    format_queries = []

    # Global: centered + wrapped
    global_format = dict(
        horizontalAlignment="CENTER",
        verticalAlignment="MIDDLE",
        wrapStrategy="WRAP",
    )
    format_queries.append(dict(range=global_range, format=global_format))

    # Bold: benchmark_id and primary metric columns (fit_time, predict_time,
    # ram_peak, vram_peak)
    bold_format = dict(textFormat=dict(bold=True))
    benchmark_id_col_range = (
        f"{gspread.utils.rowcol_to_a1(2, 1)}:"
        f"{gspread.utils.rowcol_to_a1(n_rows + 1, 1)}"
    )
    format_queries.append(dict(range=benchmark_id_col_range, format=bold_format))
    for metric in PRIMARY_METRIC_COLUMNS:
        col_idx = df.columns.get_loc(metric) + 1
        col_range = (
            f"{gspread.utils.rowcol_to_a1(2, col_idx)}:"
            f"{gspread.utils.rowcol_to_a1(n_rows + 1, col_idx)}"
        )
        format_queries.append(dict(range=col_range, format=bold_format))

    # Header: light yellow
    yellow_header = dict(
        backgroundColorStyle=dict(
            rgbColor=dict(red=1, green=1, blue=102 / 255, alpha=1)
        )
    )
    header_row_range = (
        f"{gspread.utils.rowcol_to_a1(1, 1)}:{gspread.utils.rowcol_to_a1(1, n_cols)}"
    )
    format_queries.append(dict(range=header_row_range, format=yellow_header))

    # Bi-color: every other benchmark_id block gets grey background
    bright_gray = dict(
        backgroundColorStyle=dict(
            rgbColor=dict(red=232 / 255, green=233 / 255, blue=235 / 255, alpha=1)
        )
    )
    benchmark_ids = df[BENCHMARK_ID_NAME]
    benchmark_ids_ending_idx = (
        np.where((benchmark_ids.shift() != benchmark_ids).values[1:])[0] + 2
    )
    for start, end in zip_longest(*(iter(benchmark_ids_ending_idx),) * 2):
        row_range = (
            f"{gspread.utils.rowcol_to_a1(start + 1, 1)}:"
            f"{gspread.utils.rowcol_to_a1(end or (n_rows + 1), n_cols)}"
        )
        format_queries.append(dict(range=row_range, format=bright_gray))

    # Apply all formats
    worksheet.batch_format(format_queries)

    # Auto-resize
    worksheet.columns_auto_resize(0, n_cols - 1)
    worksheet.rows_auto_resize(0, n_rows)


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    argparser = ArgumentParser(
        description=(
            "Consolidate TabICL benchopt parquet results into a CSV or sync "
            "to a Google Spreadsheet with formatted blocks."
        )
    )
    argparser.add_argument(
        "benchmark_files",
        nargs="+",
        help="benchopt parquet files to consolidate",
    )
    argparser.add_argument(
        "--sync-to-gspread",
        action="store_true",
        help="Sync to a Google Spreadsheet (requires --gspread-url and "
        "--gspread-auth-key).",
    )
    argparser.add_argument(
        "--gspread-url",
        help="URL to a Google Spreadsheet.",
    )
    argparser.add_argument(
        "--gspread-auth-key",
        help="Path to a json service account key for gspread. The file is "
        "passed directly to gspread and never read by this script.",
    )

    args = argparser.parse_args()

    df = _build_table(args.benchmark_files)

    if args.sync_to_gspread:
        if not args.gspread_url:
            raise ValueError("--gspread-url is required with --sync-to-gspread.")
        if not args.gspread_auth_key:
            raise ValueError("--gspread-auth-key is required with --sync-to-gspread.")
        _gspread_sync(df, args.gspread_url, args.gspread_auth_key)
        print(f"Synced {len(df)} rows to Google Sheets.")
    else:
        _df_to_csv(df, sys.stdout)

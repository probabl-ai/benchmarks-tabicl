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
RAM_PEAK = "RAM peak (GB)"
VRAM_PEAK = "VRAM peak (GB)"
RAM_PEAK_FIT = "RAM peak fit (GB)"
VRAM_PEAK_FIT = "VRAM peak fit (GB)"
RAM_PEAK_PREDICT = "RAM peak predict (GB)"
VRAM_PEAK_PREDICT = "VRAM peak predict (GB)"
FIT_NB_BATCHES_TF_COL = "Nb batches fit tf_col"
FIT_NB_BATCHES_TF_ROW = "Nb batches fit tf_row"
FIT_NB_BATCHES_TF_ICL = "Nb batches fit tf_icl"
PREDICT_NB_BATCHES_TF_COL = "Nb batches predict tf_col"
PREDICT_NB_BATCHES_TF_ROW = "Nb batches predict tf_row"
PREDICT_NB_BATCHES_TF_ICL = "Nb batches predict tf_icl"
ACCURACY = "Accuracy"
LOG_LOSS = "Log loss"
RMSE = "RMSE"
R2 = "R2"
NB_ESTIMATORS = "Nb estimators"
BATCH_SIZE = "Batch size"
KV_CACHE = "KV cache"
OFFLOAD_MODE = "Offload mode"
VRAM_AVAILABLE = "Available Device Memory (MiB)"
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

# The parquet stores peaks in MiB (bytes / 1024**2). The table shows decimal
# GB (1 GB = 10**9 bytes), so one MiB is 1024**2 / 10**9 GB. The simulated
# budget is left in MiB: it is a user-supplied setting and must keep its round
# values (450, 900, 12000, ...).
MIB_TO_GB = 1024**2 / 1e9
MIB_TO_GB_COLUMNS = [
    RAM_PEAK,
    VRAM_PEAK,
    RAM_PEAK_FIT,
    VRAM_PEAK_FIT,
    RAM_PEAK_PREDICT,
    VRAM_PEAK_PREDICT,
]

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
    "objective_ram_peak_fit_mb": RAM_PEAK_FIT,
    "objective_vram_peak_fit_mb": VRAM_PEAK_FIT,
    "objective_ram_peak_predict_mb": RAM_PEAK_PREDICT,
    "objective_vram_peak_predict_mb": VRAM_PEAK_PREDICT,
    "objective_fit_nb_batches_tf_col": FIT_NB_BATCHES_TF_COL,
    "objective_fit_nb_batches_tf_row": FIT_NB_BATCHES_TF_ROW,
    "objective_fit_nb_batches_tf_icl": FIT_NB_BATCHES_TF_ICL,
    "objective_predict_nb_batches_tf_col": PREDICT_NB_BATCHES_TF_COL,
    "objective_predict_nb_batches_tf_row": PREDICT_NB_BATCHES_TF_ROW,
    "objective_predict_nb_batches_tf_icl": PREDICT_NB_BATCHES_TF_ICL,
    "objective_accuracy": ACCURACY,
    "objective_log_loss": LOG_LOSS,
    "objective_rmse": RMSE,
    "objective_r2": R2,
    "p_solver_n_estimators": NB_ESTIMATORS,
    "p_solver_batch_size": BATCH_SIZE,
    "p_solver_kv_cache": KV_CACHE,
    "p_solver_offload_mode": OFFLOAD_MODE,
    "p_solver_available_memory": VRAM_AVAILABLE,
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
    RAM_PEAK_FIT,
    VRAM_PEAK_FIT,
    RAM_PEAK_PREDICT,
    VRAM_PEAK_PREDICT,
]

# Per-run batching observations (not averaged across repetitions, not metric
# columns: they are integers, or empty when the stage did not run).
OBSERVATION_COLUMNS = [
    FIT_NB_BATCHES_TF_COL,
    FIT_NB_BATCHES_TF_ROW,
    FIT_NB_BATCHES_TF_ICL,
    PREDICT_NB_BATCHES_TF_COL,
    PREDICT_NB_BATCHES_TF_ROW,
    PREDICT_NB_BATCHES_TF_ICL,
]

SANITY_METRIC_COLUMNS = [
    ACCURACY,
    LOG_LOSS,
    RMSE,
    R2,
]

NUMERIC_METRIC_COLUMNS = PRIMARY_METRIC_COLUMNS + SANITY_METRIC_COLUMNS

# Other solver parameter columns (part of the grouping key for averaging +
# dedup).
SOLVER_PARAM_COLUMNS = [
    DEVICE,
    VRAM_AVAILABLE,
    KV_CACHE,
    OFFLOAD_MODE,
    BATCH_SIZE,
    NB_JOBS,
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
# Shared by the CSV and the sheet: a plain concatenation of the top-level
# lists below. The left frame (frozen in the sheet) is the workload identity;
# the right frame starts with the headline metrics, then the solver parameters
# (device and memory budget first), the batch counts and provenance.
LEFT_FRAME_COLUMNS = [BENCHMARK_ID_NAME] + BENCHMARK_DEFINING_COLUMNS

TABLE_DISPLAY_ORDER = (
    LEFT_FRAME_COLUMNS
    + PRIMARY_METRIC_COLUMNS
    + SOLVER_PARAM_COLUMNS
    + OBSERVATION_COLUMNS
    + PROVENANCE_COLUMNS
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

    # ``rename`` ignores keys that are absent from the frame, so columns a
    # given parquet does not carry are simply left alone.
    df = df.rename(columns=PARQUET_TABLE_DISPLAY_MAPPING)

    for col in MIB_TO_GB_COLUMNS:
        if col in df.columns:
            df[col] = df[col] * MIB_TO_GB

    return df


def _aggregate_repetitions(df):
    """Average numeric metrics across repetitions (idx_rep).

    Groups by (benchmark-defining + solver params + run context + provenance).
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
    # `copy=False` is the modern default under pandas Copy-on-Write; pass it
    # explicitly to avoid a deprecation warning on older pandas.
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

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
    """Round-trip through CSV to normalize None/NaN/empty-string values.

    Float NaN in numeric metric columns becomes the string ``"nan"`` (via
    ``float_format_fn``). ``None`` in object columns (e.g. ``GPU name`` for
    CPU runs) becomes empty string ``""``. Any remaining float NaN (object
    columns that come back as NaN) is filled with ``""`` so the result is
    JSON-serializable for the gspread upload and consistent with the CSV.
    """
    buf = BytesIO()
    _df_to_csv(df, buf)
    buf.seek(0)
    df = pd.read_csv(buf, keep_default_na=False, na_values=[""])
    df = df[TABLE_DISPLAY_ORDER]
    # Object columns with None round-trip as float NaN; fill with "" (empty
    # string). Metric "nan" values are already strings, so untouched here.
    df = df.fillna("")
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
        if col not in df.columns:
            continue
        # Aggregated columns are object dtype (a mix of "mean±std" strings and
        # bare floats for zero-variance cells); format the bare floats too,
        # otherwise pandas writes them at full float precision.
        df[col] = df[col].map(
            lambda v: float_format_fn(v) if isinstance(v, (float, np.floating)) else v
        )
    df.to_csv(target, index=False, mode="a", date_format=DATES_FORMAT)


# --- spreadsheet view -------------------------------------------------------
#
# The sheet is read and filtered by hand: peaks are already in decimal GB
# (converted at load), the budget stays in MiB, every number carries three
# significant digits, and the ± std is shown only above 10% of the mean.

# Display precision: three significant digits, and a ± only when the std
# exceeds 10% of the mean.
_SIG_DIGITS = 3
STD_DISPLAY_THRESHOLD = 0.10


def _round_sig(value, digits=_SIG_DIGITS):
    """Round ``value`` to ``digits`` significant digits."""
    if value == 0 or not np.isfinite(value):
        return value
    return round(value, digits - int(np.floor(np.log10(abs(value)))) - 1)


def _fmt_sig(value):
    """Format a number with at most three significant digits, no exponent.

    Anything below ``1e-4`` is below the resolution of these measurements;
    printing ``2.2e-06`` would be noise, so it is shown as ``0``.
    """
    value = float(value)
    if abs(value) < 1e-4:
        return "0"
    value = _round_sig(value)
    return "0" if value == 0 else f"{value:g}"


def _reformat_metric(value):
    """Round a ``mean±std`` cell for the spreadsheet.

    The ± std is shown only when it exceeds ``STD_DISPLAY_THRESHOLD`` of the
    mean; below that it is visual noise.
    """
    text = str(value).strip()
    if text in ("", "nan"):
        return value
    parts = text.split("±")
    try:
        mean = float(parts[0])
    except ValueError:
        return value
    shown = _fmt_sig(mean)
    if len(parts) == 1:
        return shown
    try:
        std = float(parts[1])
    except ValueError:
        return shown
    if mean != 0 and std > STD_DISPLAY_THRESHOLD * abs(mean):
        return f"{shown}±{_fmt_sig(std)}"
    return shown


def _gspread_view(df):
    """Return the spreadsheet-friendly view of the canonical table.

    The units and column order are the canonical ones; only the numeric
    formatting changes (three significant digits, ± above 10% of the mean).
    """
    out = df.copy()

    for col in [*PRIMARY_METRIC_COLUMNS, *SANITY_METRIC_COLUMNS, VRAM_AVAILABLE]:
        if col in out.columns:
            out[col] = out[col].map(_reformat_metric)

    for col in TABLE_DISPLAY_ORDER:
        if col not in out.columns:
            out[col] = ""
    return out[TABLE_DISPLAY_ORDER]


def _gspread_sync(df, gspread_url, gspread_auth_key):
    """Sync the dataframe to a Google Spreadsheet with formatting."""
    import gspread

    df = _gspread_view(df)

    n_rows, n_cols = df.shape
    pivot_col = len(LEFT_FRAME_COLUMNS)

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

    # Upload values (the df is already sanitized by _sanitize_df_with_tocsv:
    # metric NaN → "nan" strings, object-column None → "" — no float NaN left).
    worksheet.update(
        values=[df.columns.values.tolist()] + df.values.tolist(), range_name="A1"
    )

    # Set filter
    worksheet.set_basic_filter(1, 1, n_rows + 1, n_cols)

    # Freeze: header row + left frame (run identity)
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

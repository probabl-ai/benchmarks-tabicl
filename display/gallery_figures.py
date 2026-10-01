"""Sphinx-gallery figures for the TabICL benchmark.

A single Plotly figure ("Figure 1") that tells the KV-cache story with the
minimal number of plots. It is a 3×1 grid; each row sweeps one dataset axis
(number of train rows, number of features, test-set size) and plots, for both
KV-cache settings (OFF / ON):

    * predict time       (dashed)
    * fit + predict time (solid)

Each ``build_figX`` returns a plain ``plotly.graph_objects.Figure`` so it can be
embedded in a sphinx-gallery example (``fig.write_html("example.html")``) or
served by the display app. Column names are reused from
``consolidate_result_csv`` (single source of truth for naming).

Run directly to (re)generate standalone HTML previews::

    PYTHONPATH=.:display python display/gallery_figures.py [--out DIR]

Data is read from ``results/results.csv`` (the canonical consolidated artifact:
one row per benchmark × solver-config × provenance, single repetition).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from consolidate_result_csv import (
    DEVICE,
    FIT_TIME,
    KV_CACHE,
    NB_ESTIMATORS,
    NB_FEATURES,
    NB_TEST_SAMPLES,
    NB_TRAIN_SAMPLES,
    OFFLOAD_MODE,
    PREDICT_TIME,
    RAM_PEAK,
    VRAM_PEAK,
    VRAM_PEAK_FIT,
    VRAM_PEAK_PREDICT,
)

# ---------------------------------------------------------------------------
# Visual identity (kept from the previous version)
# ---------------------------------------------------------------------------

# Kv-cache comparison — Okabe-Ito colorblind-safe palette.
COLOR_CACHE_OFF = "#009E73"  # bluish green
COLOR_CACHE_ON = "#D55E00"  # vermillion
MARKER_CACHE_OFF = "circle"
MARKER_CACHE_ON = "square"

# Offload comparison (Figure 3) — Okabe-Ito colorblind-safe palette, one color
# per device x offload config (no color reused across meanings).
COLOR_GPU_GPU = "#0072B2"  # blue         — GPU, offload=gpu
COLOR_GPU_CPU = "#56B4E9"  # sky blue     — GPU, offload=cpu
COLOR_GPU_DISK = "#CC79A7"  # reddish purple— GPU, offload=disk
COLOR_CPU_CPU = "#E69F00"  # orange       — CPU, offload=cpu
COLOR_CPU_DISK = "#D55E00"  # vermillion   — CPU, offload=disk

# Figure 3 bar metric colors — Okabe-Ito, distinct from each other and from
# the offload-config colors above.
COLOR_BAR_TIME = "#000000"  # black   — predict time
COLOR_BAR_RAM = "#0072B2"  # blue    — RAM peak
COLOR_BAR_VRAM = "#D55E00"  # vermillion — VRAM peak
MARKER_GPU_GPU = "circle"
MARKER_GPU_CPU = "diamond"
MARKER_GPU_DISK = "square"
MARKER_CPU_CPU = "triangle-up"
MARKER_CPU_DISK = "triangle-down"

# ---------------------------------------------------------------------------
# Margin profiles. Two rendering targets share the same figure builders:
#   * ``default`` — generous margins for direct/standalone HTML viewing.
#   * ``gallery``  — tight margins tuned for the narrow sphinx-gallery column.
# The ``<br>`` title/subtitle wrapping and the annotation-based title
# positioning are correctness (identical in both); only the horizontal margins,
# inter-subplot spacing, and y-axis title standoff differ. The plot area is
# fixed at 240 px tall in both, so vertical proportions never change.
# ---------------------------------------------------------------------------
_DEFAULT_LR = (80, 80)  # generous left/right for standalone viewing
_GALLERY_LR = (20, 5)  # tight for the narrow gallery column
_DEFAULT_YSTANDOFF = None  # plotly default
_GALLERY_YSTANDOFF = 6


def _hspacing(fig_key: str, gallery: bool) -> float:
    """Inter-subplot horizontal spacing per figure kind and target."""
    if fig_key == "fig3":
        return 0.08 if gallery else 0.12
    return 0.04 if gallery else 0.07


# Run-label filters
LABEL_SCALING = "rows & cols scaling"
LABEL_OFFLOAD = "offload modes GPU vs CPU"
LABEL_OFFLOAD_LARGE_NTEST = "GPU offload at large n_test"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Return a lighter shade of a hex colour (mix with white by ``factor``)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _parse_mean(value):
    """Extract the mean from a ``mean±std`` repetition-average string.

    Returns the value as-is if already numeric/empty.
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    if "±" in s:
        s = s.split("±")[0].strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_results(csv_path: str | Path = "results/results.csv") -> pd.DataFrame:
    """Load the consolidated results CSV with coerced dtypes.

    Numeric metric columns may be stored as ``mean±std`` strings when the
    consolidation averaged multiple repetitions — the mean is extracted so the
    figures receive plain floats.
    """
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])

    for col in (KV_CACHE,):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map({"True": True, "False": False, "true": True, "false": False})
            )

    for col in (
        FIT_TIME,
        PREDICT_TIME,
        RAM_PEAK,
        VRAM_PEAK,
        VRAM_PEAK_FIT,
        VRAM_PEAK_PREDICT,
    ):
        if col in df.columns:
            df[col] = df[col].map(_parse_mean)

    return df


def _agg(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """Group by ``by`` and take the mean of ``metric``."""
    return df.groupby(by, dropna=False, sort=False)[metric].mean().reset_index()


# ---------------------------------------------------------------------------
# Figure 1 — KV cache: predict & fit+predict vs rows, cols, test-set size
# ---------------------------------------------------------------------------


def build_fig1_kv_cache(
    df: pd.DataFrame, gallery: bool = False, mobile: bool = False
) -> go.Figure:
    """Predict and fit+predict time vs dataset size (3 axes), KV cache OFF vs ON.

    Layout: 1 row × 3 cols. Each cell plots four lines:

        * predict       — kv OFF (green, dashed)
        * predict       — kv ON  (vermillion, dashed)
        * fit+predict   — kv OFF (green, solid)
        * fit+predict   — kv ON  (vermillion, solid)

      * row 1: vs n_train     (rows sweep; n_features=100, n_test=2000)
      * row 2: vs n_features  (cols sweep; n_train=1000,  n_test=2000)
      * row 3: vs n_test      (rows sweep; n_train=4000,  n_features=100)

    n_estimators=4, classification, 10 classes, cuda, offload=gpu.
    Log axes.
    """
    sub = df[df["Run label"].str.contains(LABEL_SCALING, na=False)].copy()
    sub = sub[sub[NB_ESTIMATORS] == 4]
    sub["Total (fit + predict)"] = sub[FIT_TIME] + sub[PREDICT_TIME]

    # (row, x-axis column, filter dict, x-axis title, subplot title)
    rows_spec = [
        (
            1,
            NB_TRAIN_SAMPLES,
            {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
            "Number of train rows",
            "Time vs number of train rows<br>(n_features=100, n_test=2000)",
        ),
        (
            2,
            NB_FEATURES,
            {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
            "Number of features",
            "Time vs number of features<br>(n_train=1000, n_test=2000)",
        ),
        (
            3,
            NB_TEST_SAMPLES,
            {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
            "Test-set size",
            "Time vs test-set size<br>(n_train=4000, n_features=100)",
        ),
    ]

    if mobile:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            vertical_spacing=0.18,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            horizontal_spacing=_hspacing("fig12", gallery),
        )

    # (kv value, color, marker, kv_name)
    kv_settings = [
        (False, COLOR_CACHE_OFF, MARKER_CACHE_OFF, "OFF"),
        (True, COLOR_CACHE_ON, MARKER_CACHE_ON, "ON"),
    ]

    # (metric column, dash, width, metric_name)
    metrics = [
        (PREDICT_TIME, "dash", 2.0, "predict"),
        ("Total (fit + predict)", "solid", 2.5, "fit+predict"),
    ]

    for col_idx, x_col, filt, x_title, _ in rows_spec:
        cell = sub
        for k, v in filt.items():
            cell = cell[cell[k] == v]

        r, c = (col_idx, 1) if mobile else (1, col_idx)
        for kv, color, marker, kv_name in kv_settings:
            d = cell[cell[KV_CACHE] == kv].sort_values(x_col)
            for metric, dash, lw, metric_name in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=d[x_col],
                        y=d[metric],
                        name=f"kv {kv_name} — {metric_name}",
                        legendgroup=f"kv {kv_name} — {metric_name}",
                        showlegend=(col_idx == 1),
                        mode="lines+markers",
                        line=dict(color=color, width=lw, dash=dash),
                        marker=dict(symbol=marker, size=7, color=color),
                        hovertemplate=(
                            f"{x_title}=%{{x}}<br>{metric_name}=%{{y:.3g}} s"
                            f"<extra>kv {kv_name}</extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

        fig.update_xaxes(type="log", title_text=x_title, row=r, col=c)
        fig.update_yaxes(type="log", row=r, col=c)

    fig.update_yaxes(title_text="Time (s)", row=1, col=1)

    _finalize(
        fig,
        title="Figure 1 — KV cache: predict and fit+predict time vs dataset size",
        subtitle="GPU (NVIDIA L4), classification, 10 classes, "
        "n_estimators=4, offload=gpu.<br>"
        "Lines: predict (dashed), fit+predict (solid); "
        "kv cache OFF (green) vs ON (vermillion). Log axes.",
        height=560,
        legend_title="KV cache × metric",
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Peak VRAM vs dataset size (kv cache OFF vs ON)
# ---------------------------------------------------------------------------


def build_fig2_vram(
    df: pd.DataFrame, gallery: bool = False, mobile: bool = False
) -> go.Figure:
    """Peak VRAM vs three dataset-size axes, KV cache OFF vs ON.

    Layout: 1 row × 3 cols. Each cell plots three lines:

        * predict       — kv OFF (green, dashed)
        * predict       — kv ON  (vermillion, dashed)
        * fit           — kv ON  (vermillion, solid)

    ``kv OFF — fit`` is omitted: it is constant at ~110 MB across all axes
    (the estimator's small footprint, no kv cache built) and would just sit
    flat at the bottom of the log plot.

    Note: the kv cache is per-estimator — each estimator in the ensemble uses
    shifted columns, so cached context does not carry over across estimators.
    n_estimators is therefore fixed (to 4) here; sweeping it just scales both
    the cache cost and benefit linearly without amortization.

    Phase definitions (see ``benchmark_utils.base_solver``):

        * ``fit``         = ``VRAM peak fit (MB)``  — incremental peak within
          the fit window only (the kv cache build lives here).
        * ``predict``     = ``VRAM peak predict (MB)`` — absolute footprint
          during predict = (VRAM retained at end of fit) + (incremental peak
          within the predict window). This is the real-workload inference
          footprint, including the resident kv cache.

      * panel 1: vs n_train     (rows sweep; n_features=100, n_test=2000)
      * panel 2: vs n_features  (cols sweep; n_train=1000,  n_test=2000)
      * panel 3: vs n_test      (rows sweep; n_train=4000,  n_features=100)

    n_estimators=4, classification, 10 classes, cuda, offload=gpu. Log axes.
    """
    sub = df[df["Run label"].str.contains(LABEL_SCALING, na=False)].copy()
    sub = sub[sub[NB_ESTIMATORS] == 4]

    # (col, x-axis column, filter dict, x-axis title, subplot title)
    rows_spec = [
        (
            1,
            NB_TRAIN_SAMPLES,
            {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
            "Number of train rows",
            "VRAM vs number of train rows<br>(n_features=100, n_test=2000)",
        ),
        (
            2,
            NB_FEATURES,
            {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
            "Number of features",
            "VRAM vs number of features<br>(n_train=1000, n_test=2000)",
        ),
        (
            3,
            NB_TEST_SAMPLES,
            {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
            "Test-set size",
            "VRAM vs test-set size<br>(n_train=4000, n_features=100)",
        ),
    ]

    if mobile:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            vertical_spacing=0.18,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            horizontal_spacing=_hspacing("fig12", gallery),
        )

    # Per-kv metric sets. ``kv OFF — fit`` is omitted: it is constant at
    # ~110 MB across all axes (the estimator's small footprint, no kv cache
    # built), so it would just sit flat at the bottom of the log plot.
    kv_metrics = [
        # (kv value, color, marker, kv_name, [(metric, dash, lw, metric_name), ...])
        (
            False,
            COLOR_CACHE_OFF,
            MARKER_CACHE_OFF,
            "OFF",
            [
                (VRAM_PEAK_PREDICT, "dash", 2.5, "predict"),
            ],
        ),
        (
            True,
            COLOR_CACHE_ON,
            MARKER_CACHE_ON,
            "ON",
            [
                (VRAM_PEAK_PREDICT, "dash", 2.5, "predict"),
                (VRAM_PEAK_FIT, "solid", 3.5, "fit"),
            ],
        ),
    ]

    for col_idx, x_col, filt, x_title, _ in rows_spec:
        cell = sub
        for k, v in filt.items():
            cell = cell[cell[k] == v]

        r, c = (col_idx, 1) if mobile else (1, col_idx)
        for kv, color, marker, kv_name, metrics in kv_metrics:
            d = cell[cell[KV_CACHE] == kv].sort_values(x_col)
            for metric, dash, lw, metric_name in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=d[x_col],
                        y=d[metric],
                        name=f"kv {kv_name} — {metric_name}",
                        legendgroup=f"kv {kv_name} — {metric_name}",
                        showlegend=(col_idx == 1),
                        mode="lines+markers",
                        line=dict(color=color, width=lw, dash=dash),
                        marker=dict(symbol=marker, size=10, color=color),
                        hovertemplate=(
                            f"{x_title}=%{{x}}<br>VRAM {metric_name}=%{{y:.0f}} MB"
                            f"<extra>kv {kv_name}</extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

        fig.update_xaxes(type="log", title_text=x_title, row=r, col=c)
        fig.update_yaxes(type="log", row=r, col=c)

    fig.update_yaxes(title_text="VRAM peak (MB)", row=1, col=1)

    _finalize(
        fig,
        title="Figure 2 — KV cache: peak VRAM (fit, predict) vs dataset size",
        subtitle="GPU (NVIDIA L4), classification, 10 classes, "
        "n_estimators=4, offload=gpu.<br>"
        "Lines: fit (solid), predict (dashed); "
        "predict = resident fit state + predict peak.<br>"
        "kv OFF — fit is omitted (constant ~110 MB, negligible).<br>"
        "kv OFF (green) vs ON (vermillion). Log axes.",
        height=560,
        legend_title="KV cache × phase",
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Offload comparison: GPU vs CPU, offload modes
# ---------------------------------------------------------------------------


def build_fig3_offload(
    df: pd.DataFrame, gallery: bool = False, mobile: bool = False
) -> go.Figure:
    """Offload comparison: GPU vs CPU (left) and GPU offload at large n_test (right).

    Layout: 1 row × 2 cols.
      * Left panel (Command B): grouped bars — X = device × offload mode,
        Y (log) = predict time. Configs: GPU-gpu, GPU-cpu, GPU-disk, CPU.
        n_train=4000, n_test=200, n_features=200, n_est=4, kv=False.
      * Right panel (Command C): grouped bars — X = GPU offload mode,
        Y (log) = predict time. GPU only, n_train=4000, n_test=10000,
        n_features=200, n_est=4, kv=False.

    Both panels also show RAM and VRAM as a second grouped bar chart below
    each predict-time chart (handled via a 2-row subplot layout).

    Actually: to keep it simple and consistent with the previous version, each
    panel is a single bar chart for predict time, and a second bar chart for
    memory (RAM + VRAM). We use a 2×2 grid:
      row 1: predict time (left=B, right=C)
      row 2: peak memory (left=B, right=C)
    """
    # --- Left panel data: Command B (n_test=200, GPU+CPU) ---
    sub_b = df[df["Run label"].str.contains(LABEL_OFFLOAD, na=False)].copy()
    sub_b = sub_b[(sub_b[NB_TRAIN_SAMPLES] == 4000) & (sub_b[NB_ESTIMATORS] == 4)]

    gpu_configs = [
        ("cuda", "gpu", "GPU — offload=gpu"),
        ("cuda", "cpu", "GPU — offload=cpu"),
        ("cuda", "disk", "GPU — offload=disk"),
    ]
    x_labels_b = [c[2] for c in gpu_configs] + ["CPU"]

    def _metric(rows, metric):
        vals = rows[metric].map(_parse_mean)
        return float(vals.mean()) if len(vals) else None

    cpu_rows_b = sub_b[(sub_b[DEVICE] == "cpu") & (sub_b[OFFLOAD_MODE] == "cpu")]

    # --- Right panel data: Command C (n_test=10000, GPU only) ---
    sub_c = df[df["Run label"].str.contains(LABEL_OFFLOAD_LARGE_NTEST, na=False)].copy()
    sub_c = sub_c[(sub_c[NB_TRAIN_SAMPLES] == 4000) & (sub_c[NB_ESTIMATORS] == 4)]

    gpu_only_configs = [
        ("cuda", "gpu", "GPU — offload=gpu"),
        ("cuda", "cpu", "GPU — offload=cpu"),
        ("cuda", "disk", "GPU — offload=disk"),
    ]
    x_labels_c = [c[2] for c in gpu_only_configs]

    if mobile:
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "n_test=200<br>(GPU vs CPU, n_train=4000)",
                "n_test=10 000<br>(GPU only, n_train=4000)",
            ],
            vertical_spacing=0.36,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "n_test=200<br>(GPU vs CPU, n_train=4000)",
                "n_test=10 000<br>(GPU only, n_train=4000)",
            ],
            horizontal_spacing=_hspacing("fig3", gallery),
        )

    # --- Left panel: predict time + memory (3 traces) ---
    # Predict time
    ys_time_b = []
    for device, offload, _ in gpu_configs:
        row = sub_b[(sub_b[DEVICE] == device) & (sub_b[OFFLOAD_MODE] == offload)]
        ys_time_b.append(_metric(row, PREDICT_TIME))
    ys_time_b.append(_metric(cpu_rows_b, PREDICT_TIME))
    fig.add_trace(
        go.Bar(
            x=x_labels_b,
            y=ys_time_b,
            name="Predict time",
            legendgroup="time",
            showlegend=True,
            marker_color=COLOR_BAR_TIME,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>predict=%{y:.3g} s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # RAM
    ys_ram_b = []
    for device, offload, _ in gpu_configs:
        row = sub_b[(sub_b[DEVICE] == device) & (sub_b[OFFLOAD_MODE] == offload)]
        ys_ram_b.append(_metric(row, RAM_PEAK))
    ys_ram_b.append(_metric(cpu_rows_b, RAM_PEAK))
    fig.add_trace(
        go.Bar(
            x=x_labels_b,
            y=ys_ram_b,
            name="RAM peak",
            legendgroup="ram",
            showlegend=True,
            marker_color=COLOR_BAR_RAM,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>RAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # VRAM
    ys_vram_b = []
    for device, offload, _ in gpu_configs:
        row = sub_b[(sub_b[DEVICE] == device) & (sub_b[OFFLOAD_MODE] == offload)]
        ys_vram_b.append(_metric(row, VRAM_PEAK))
    ys_vram_b.append(_metric(cpu_rows_b, VRAM_PEAK))
    fig.add_trace(
        go.Bar(
            x=x_labels_b,
            y=ys_vram_b,
            name="VRAM peak",
            legendgroup="vram",
            showlegend=True,
            marker_color=COLOR_BAR_VRAM,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>VRAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Right panel: predict time + memory (3 traces, GPU only) ---
    r2, c2 = (2, 1) if mobile else (1, 2)
    # Predict time
    ys_time_c = []
    for device, offload, _ in gpu_only_configs:
        row = sub_c[(sub_c[DEVICE] == device) & (sub_c[OFFLOAD_MODE] == offload)]
        ys_time_c.append(_metric(row, PREDICT_TIME))
    fig.add_trace(
        go.Bar(
            x=x_labels_c,
            y=ys_time_c,
            name="Predict time",
            legendgroup="time",
            showlegend=False,
            marker_color=COLOR_BAR_TIME,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>predict=%{y:.3g} s<extra></extra>",
        ),
        row=r2,
        col=c2,
    )
    # RAM
    ys_ram_c = []
    for device, offload, _ in gpu_only_configs:
        row = sub_c[(sub_c[DEVICE] == device) & (sub_c[OFFLOAD_MODE] == offload)]
        ys_ram_c.append(_metric(row, RAM_PEAK))
    fig.add_trace(
        go.Bar(
            x=x_labels_c,
            y=ys_ram_c,
            name="RAM peak",
            legendgroup="ram",
            showlegend=False,
            marker_color=COLOR_BAR_RAM,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>RAM=%{y:.0f} MB<extra></extra>",
        ),
        row=r2,
        col=c2,
    )
    # VRAM
    ys_vram_c = []
    for device, offload, _ in gpu_only_configs:
        row = sub_c[(sub_c[DEVICE] == device) & (sub_c[OFFLOAD_MODE] == offload)]
        ys_vram_c.append(_metric(row, VRAM_PEAK))
    fig.add_trace(
        go.Bar(
            x=x_labels_c,
            y=ys_vram_c,
            name="VRAM peak",
            legendgroup="vram",
            showlegend=False,
            marker_color=COLOR_BAR_VRAM,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>VRAM=%{y:.0f} MB<extra></extra>",
        ),
        row=r2,
        col=c2,
    )

    if mobile:
        for r in (1, 2):
            fig.update_xaxes(title_text="Device × offload mode", row=r, col=1)
            fig.update_yaxes(type="log", row=r, col=1)
    else:
        for c in (1, 2):
            fig.update_xaxes(title_text="Device × offload mode", row=1, col=c)
            fig.update_yaxes(type="log", row=1, col=c)
    fig.update_yaxes(title_text="Predict time (s) / Memory (MB)", row=1, col=1)

    _finalize(
        fig,
        title=(
            "Figure 3 — Offload comparison: GPU vs CPU (left) and large n_test (right)"
        ),
        subtitle="Classification, 10 classes, n_estimators=4, "
        "kv_cache=False, n_features=200, n_train=4000.<br>"
        "Left: n_test=200, GPU vs CPU (offload no-op on CPU).<br>"
        "Right: n_test=10 000, GPU only — large output tensors make "
        "offload modes save ~2.5 GB VRAM.<br>"
        "Bars: predict time (black), RAM (blue), VRAM (vermillion). Log y.",
        height=560,
        legend_title="Metric",
        gallery=gallery,
        extra_bottom=60,
        plot_area_px=400 if mobile else 240,
        mobile=mobile,
        legend_offset_px=68 if mobile else None,
    )
    return fig


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------


def _finalize(
    fig: go.Figure,
    title: str,
    subtitle: str,
    height: int,
    legend_title: str,
    gallery: bool = False,
    extra_bottom: int = 0,
    plot_area_px: int = 240,
    mobile: bool = False,
    legend_offset_px: int | None = None,
) -> None:
    # Keep the plot area IDENTICAL to the HEAD baseline (240 px tall) so the
    # plot proportions never change. The figure title block (bold title +
    # ``<br>``-wrapped subtitle) is rendered as a *layout annotation* pinned
    # at the top of the margin band with ``yanchor="top"`` (grows down),
    # because plotly's ``layout.title`` is capped at y=1 (the domain top) and
    # a multi-line block would grow down from there into the subplot titles.
    #
    # The top margin must hold: the figure title block + a gap + the subplot
    # titles (which grow up from y=1). A 4-line subtitle needs more band than
    # a 2-line one, so ``t`` scales with the subtitle line count. To keep the
    # plot area fixed at 240 px while ``t`` changes, ``height`` is adjusted by
    # the same delta (height = 240 + t + b). So:
    #   - 2-line subtitle -> smaller t, smaller total height (less gap).
    #   - 4-line subtitle -> larger t, larger total height (more gap).
    #
    # Horizontal margins and y-axis title standoff depend on the target:
    # ``gallery`` tightens them for the narrow sphinx-gallery column, while
    # the default profile uses generous margins for direct/standalone viewing.
    n_subtitle_lines = subtitle.count("<br>") + 1
    # Per-line budgets tuned to plotly's actual rendered text metrics (font
    # 18 bold title ~30px/line, 13px subtitle ~22px/line, two-line subplot
    # title at font 14 ~48px). The gap between the title block and the subplot
    # titles scales up slightly with the subtitle line count so 4-row
    # subtitles get a bit more breathing room than 2-row ones.
    title_block_px = 30 + n_subtitle_lines * 22
    subplot_title_px = 48
    gap_px = 16 + n_subtitle_lines * 2
    # Bottom margin + legend position depend on the target. In gallery mode
    # the legend sits just below the plot and grows *downward* into the
    # bottom margin (anchored at top). This way, when the legend wraps to
    # multiple lines on a narrow screen, the extra lines grow down into the
    # margin (harmless) instead of up into the plot. In default mode the
    # legend stays at a comfortable mid-band position with a generous bottom
    # margin.
    if gallery:
        bottom_margin = 90 + extra_bottom
        # Legend sits a fixed *pixel* offset below the plot domain (just past
        # the x-axis tick labels + title), converted to paper coords via the
        # plot area height. This keeps the pixel gap consistent across
        # desktop (plot_area 240) and mobile (400-600), and pushes lower for
        # fig3 whose diagonal tick labels need ``extra_bottom`` of room.
        # ``legend_offset_px`` overrides the computed offset (used by fig3
        # mobile, where the default 45+extra_bottom sits too far below the
        # last subplot's x-axis title).
        legend_offset = (
            legend_offset_px if legend_offset_px is not None else 45 + extra_bottom
        )
        legend_y = -legend_offset / plot_area_px
        legend_yanchor = "top"
    else:
        bottom_margin = 160 + extra_bottom
        legend_y = -0.22 - extra_bottom / 240.0
        legend_yanchor = "top"
    plot_area_px = plot_area_px  # fixed per call (240 desktop, more for mobile stacks)
    top_margin = title_block_px + gap_px + subplot_title_px
    height = plot_area_px + top_margin + bottom_margin
    band_top = 1 + top_margin / plot_area_px  # paper y of the figure's top edge
    left_margin, right_margin = _GALLERY_LR if gallery else _DEFAULT_LR
    y_standoff = _GALLERY_YSTANDOFF if gallery else _DEFAULT_YSTANDOFF
    # Capture the existing (subplot-title) annotations BEFORE adding the
    # figure-title annotation, so the baseline-positioning loop below only
    # touches the subplot titles, not the figure title.
    n_subplot_anns = len(fig.layout.annotations)
    # Render the figure title block as an annotation pinned at the band top.
    fig.add_annotation(
        text=(
            f"<b>{title}</b><br>"
            f"<span style='font-size:13px;color:#6b7280'>{subtitle}</span>"
        ),
        x=0.5,
        xref="paper",
        y=band_top,
        yref="paper",
        yanchor="top",
        xanchor="center",
        showarrow=False,
        font=dict(size=18),
    )
    fig.update_layout(
        # layout.title left empty: the figure title is the annotation above.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(
            title=dict(text=f"<b>{legend_title}</b>", font_size=13),
            orientation="h",
            y=legend_y,
            yanchor=legend_yanchor,
            x=0.5,
            xanchor="center",
            font_size=12,
        ),
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
        height=height,
    )
    # Subplot titles: keep the baseline positioning (lifted +0.03, growing
    # up). Do NOT reposition them — moving them changes the plot's apparent
    # proportions. Only touch the pre-existing annotations (by index), not
    # the figure-title annotation we just appended.
    for i in range(n_subplot_anns):
        fig.layout.annotations[i].update(
            y=fig.layout.annotations[i].y + 0.03,
            yanchor="bottom",
            font=dict(size=12, color="#1f2937"),
        )
    yaxis_kwargs = dict(
        zeroline=False,
        gridcolor="rgba(0,0,0,0.10)",
        title_font=dict(size=13, color="#1f2937"),
        tickfont=dict(size=11, color="#374151"),
    )
    if y_standoff is not None:
        yaxis_kwargs["title_standoff"] = y_standoff
    fig.update_yaxes(**yaxis_kwargs)
    fig.update_xaxes(
        zeroline=False,
        gridcolor="rgba(0,0,0,0.10)",
        title_font=dict(size=13, color="#1f2937"),
        tickfont=dict(size=11, color="#374151"),
    )


# ---------------------------------------------------------------------------
# Comparison: overlay a second dataset with a visibility toggle
# ---------------------------------------------------------------------------


def build_comparison_fig(
    build_fn,
    df_base: pd.DataFrame,
    df_other: pd.DataFrame,
    base_label: str = "baseline (2.2.0)",
    other_label: str = "pr162",
    gallery: bool = False,
    mobile: bool = False,
) -> go.Figure:
    """Build a toggleable comparison figure from two datasets.

    Builds a figure identical to ``build_fn(df_base)`` but with a second
    dataset (``df_other``) overlaid as hidden traces and a toggle button to
    switch between the two.

    The two datasets must produce the same number of traces in the same
    order (the builders add one trace per structural cell, so this holds as
    long as the same figure is built from both). The toggle uses plotly
    ``updatemenus`` ``restyle`` to flip the ``visible`` flag of each set.
    """
    fig = build_fn(df_base, gallery=gallery, mobile=mobile)
    fig_other = build_fn(df_other, gallery=gallery, mobile=mobile)

    n_base = len(fig.data)
    n_other = len(fig_other.data)
    if n_other != n_base:
        raise ValueError(
            f"{build_fn.__name__}: trace-count mismatch between datasets "
            f"(base={n_base}, other={n_other}); the comparison grids must "
            f"produce the same figure structure."
        )

    # Append the second dataset's traces, hidden by default. ``add_trace``
    # deep-copies the trace, so mutating ``fig_other.data`` first is safe.
    # Traces keep their ``xaxis``/``yaxis`` assignment, so they land in the
    # correct subplots of ``fig``.
    for tr in list(fig_other.data):
        tr.update(visible=False)
        fig.add_trace(tr)

    n = n_base
    visible_base = [True] * n + [False] * n
    visible_other = [False] * n + [True] * n

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=-0.18,
                yanchor="top",
                bgcolor="rgba(0,0,0,0.04)",
                bordercolor="#9ca3af",
                font=dict(size=14, color="#1f2937"),
                buttons=[
                    dict(
                        label=base_label,
                        method="restyle",
                        args=[{"visible": visible_base}],
                    ),
                    dict(
                        label=other_label,
                        method="restyle",
                        args=[{"visible": visible_other}],
                    ),
                ],
            )
        ]
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the gallery figures (CLI entry point)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", default="results/results.csv", help="consolidated results CSV"
    )
    ap.add_argument(
        "--compare-csv",
        default=None,
        help=(
            "Second consolidated results CSV to overlay with a toggle. "
            "When set, writes <name>_comparison_<suffix>.html for each "
            "figure (the standalone figures are left untouched)."
        ),
    )
    ap.add_argument(
        "--compare-suffix",
        default="pr162",
        help="Suffix for the comparison output filenames (default: pr162).",
    )
    ap.add_argument(
        "--compare-base-label",
        default="baseline (2.2.0)",
        help="Toggle button label for the --csv dataset.",
    )
    ap.add_argument(
        "--compare-other-label",
        default="pr162",
        help="Toggle button label for the --compare-csv dataset.",
    )
    ap.add_argument(
        "--out", default="display/gallery_figures", help="output dir for HTML/embed"
    )
    ap.add_argument(
        "--gallery-html",
        action="store_true",
        help=(
            "Write gallery-tuned HTML embed snippets (<out>/embed/<name>.html) "
            "with tight margins for the narrow sphinx-gallery column, for "
            "inlining into a sphinx page via ``.. raw:: html :file:``. Uses "
            "fig.to_html(full_html=False, include_plotlyjs='cdn') so the "
            "snippet is a self-contained <div>+<script> block. "
            "Standalone/standalone-comparison HTML (the default and "
            "--compare-csv modes) always uses the default (generous) margins."
        ),
    )
    args = ap.parse_args()

    df = load_results(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builders = {
        "fig1_kv_cache": build_fig1_kv_cache,
        "fig2_memory": build_fig2_vram,
        "fig3_offload": build_fig3_offload,
    }

    if args.gallery_html:
        # Gallery-tuned HTML embed snippets for inlining into a sphinx page
        # via ``.. raw:: html :file:``. Each file contains BOTH a desktop
        # layout (1 row x N cols) and a mobile layout (N rows x 1 col),
        # wrapped in ``.plotly-desktop`` / ``.plotly-mobile`` divs so the
        # docs CSS can swap them by viewport width. Tight margins for the
        # narrow gallery column; plotly loaded from the CDN (once, in the
        # desktop snippet — the mobile snippet reuses it).
        embed_dir = out / "embed"
        embed_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in builders.items():
            desktop = fn(df, gallery=True, mobile=False)
            mobile_fig = fn(df, gallery=True, mobile=True)
            html = (
                '<div class="plotly-desktop">\n'
                + desktop.to_html(full_html=False, include_plotlyjs="cdn")
                + "\n</div>\n"
                '<div class="plotly-mobile">\n'
                + mobile_fig.to_html(full_html=False, include_plotlyjs=False)
                + "\n</div>\n"
            )
            path = embed_dir / f"{name}.html"
            path.write_text(html)
            print(f"wrote {path}")
    elif args.compare_csv is None:
        # Standalone HTML with default (generous) margins.
        for name, fn in builders.items():
            fig = fn(df)
            path = out / f"{name}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            print(f"wrote {path}")
    else:
        # Comparison HTML with default (generous) margins.
        df_other = load_results(args.compare_csv)
        for name, fn in builders.items():
            fig = build_comparison_fig(
                fn,
                df,
                df_other,
                base_label=args.compare_base_label,
                other_label=args.compare_other_label,
            )
            path = out / f"{name}_comparison_{args.compare_suffix}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

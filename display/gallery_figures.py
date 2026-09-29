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

# KV-cache comparison
COLOR_CACHE_OFF = "#6b7280"  # slate
COLOR_CACHE_ON = "#dc2626"  # crimson
MARKER_CACHE_OFF = "circle"
MARKER_CACHE_ON = "square"

# Offload comparison (Figure 3) — one color per device × offload config
COLOR_GPU_GPU = "#2563eb"    # blue   — GPU, offload=gpu
COLOR_GPU_CPU = "#0891b2"   # teal   — GPU, offload=cpu
COLOR_GPU_DISK = "#7c3aed"  # purple — GPU, offload=disk
COLOR_CPU_CPU = "#ea580c"   # orange — CPU, offload=cpu
COLOR_CPU_DISK = "#dc2626"  # crimson— CPU, offload=disk
MARKER_GPU_GPU = "circle"
MARKER_GPU_CPU = "diamond"
MARKER_GPU_DISK = "square"
MARKER_CPU_CPU = "triangle-up"
MARKER_CPU_DISK = "triangle-down"

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
    """Extract the mean from a ``mean±std`` string (consolidated repetition
    average) or return the value as-is if already numeric/empty."""
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
            df[col] = df[col].astype(str).str.strip().map(
                {"True": True, "False": False, "true": True, "false": False}
            )

    for col in (FIT_TIME, PREDICT_TIME, RAM_PEAK, VRAM_PEAK, VRAM_PEAK_FIT, VRAM_PEAK_PREDICT):
        if col in df.columns:
            df[col] = df[col].map(_parse_mean)

    return df


def _agg(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """Group by ``by`` and take the mean of ``metric``."""
    return (
        df.groupby(by, dropna=False, sort=False)[metric]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figure 1 — KV cache: predict & fit+predict vs rows, cols, test-set size
# ---------------------------------------------------------------------------

def build_fig1_kv_cache(df: pd.DataFrame) -> go.Figure:
    """Predict and fit+predict time vs three dataset-size axes, KV cache OFF
    vs ON.

    Layout: 1 row × 3 cols. Each cell plots four lines:

        * predict       — kv OFF (slate, dashed)
        * predict       — kv ON  (crimson, dashed)
        * fit+predict   — kv OFF (slate, solid)
        * fit+predict   — kv ON  (crimson, solid)

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
        (1, NB_TRAIN_SAMPLES, {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
         "Number of train rows",
         "Time vs number of train rows  (n_features=100, n_test=2000)"),
        (2, NB_FEATURES, {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
         "Number of features",
         "Time vs number of features  (n_train=1000, n_test=2000)"),
        (3, NB_TEST_SAMPLES, {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
         "Test-set size",
         "Time vs test-set size  (n_train=4000, n_features=100)"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _, _, _, _, title in rows_spec],
        horizontal_spacing=0.07,
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
                    row=1, col=col_idx,
                )

        fig.update_xaxes(type="log", title_text=x_title, row=1, col=col_idx)
        fig.update_yaxes(type="log", row=1, col=col_idx)

    fig.update_yaxes(title_text="Time (s)", row=1, col=1)

    _finalize(
        fig,
        title="Figure 1 — KV cache: predict and fit+predict time vs dataset size",
        subtitle="GPU (NVIDIA L4), classification, 10 classes, n_estimators=4, offload=gpu. "
                 "Lines: predict (dashed), fit+predict (solid); "
                 "kv cache OFF (slate) vs ON (crimson). Log axes.",
        height=560,
        legend_title="KV cache × metric",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Peak VRAM vs dataset size (kv cache OFF vs ON)
# ---------------------------------------------------------------------------

def build_fig2_vram(df: pd.DataFrame) -> go.Figure:
    """Peak VRAM vs three dataset-size axes, KV cache OFF vs ON.

    Layout: 1 row × 3 cols. Each cell plots three lines:

        * predict       — kv OFF (slate, dashed)
        * predict       — kv ON  (crimson, dashed)
        * fit           — kv ON  (crimson, solid)

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
        (1, NB_TRAIN_SAMPLES, {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
         "Number of train rows",
         "VRAM vs number of train rows  (n_features=100, n_test=2000)"),
        (2, NB_FEATURES, {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
         "Number of features",
         "VRAM vs number of features  (n_train=1000, n_test=2000)"),
        (3, NB_TEST_SAMPLES, {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
         "Test-set size",
         "VRAM vs test-set size  (n_train=4000, n_features=100)"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _, _, _, _, title in rows_spec],
        horizontal_spacing=0.07,
    )

    # Per-kv metric sets. ``kv OFF — fit`` is omitted: it is constant at
    # ~110 MB across all axes (the estimator's small footprint, no kv cache
    # built), so it would just sit flat at the bottom of the log plot.
    kv_metrics = [
        # (kv value, color, marker, kv_name, [(metric, dash, lw, metric_name), ...])
        (False, COLOR_CACHE_OFF, MARKER_CACHE_OFF, "OFF", [
            (VRAM_PEAK_PREDICT, "dash", 2.5, "predict"),
        ]),
        (True, COLOR_CACHE_ON, MARKER_CACHE_ON, "ON", [
            (VRAM_PEAK_PREDICT, "dash", 2.5, "predict"),
            (VRAM_PEAK_FIT, "solid", 3.5, "fit"),
        ]),
    ]

    for col_idx, x_col, filt, x_title, _ in rows_spec:
        cell = sub
        for k, v in filt.items():
            cell = cell[cell[k] == v]

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
                    row=1, col=col_idx,
                )

        fig.update_xaxes(type="log", title_text=x_title, row=1, col=col_idx)
        fig.update_yaxes(type="log", row=1, col=col_idx)

    fig.update_yaxes(title_text="VRAM peak (MB)", row=1, col=1)

    _finalize(
        fig,
        title="Figure 2 — KV cache: peak VRAM (fit, predict) vs dataset size",
        subtitle="GPU (NVIDIA L4), classification, 10 classes, n_estimators=4, offload=gpu. "
                 "Lines: fit (solid), predict (dashed); "
                 "predict = resident fit state + predict peak. "
                 "kv OFF — fit is omitted (constant ~110 MB, negligible). "
                 "kv OFF (slate) vs ON (crimson). Log axes.",
        height=560,
        legend_title="KV cache × phase",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Offload comparison: GPU vs CPU, offload modes
# ---------------------------------------------------------------------------

def build_fig3_offload(df: pd.DataFrame) -> go.Figure:
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
        ("cuda", "gpu",  "GPU — offload=gpu"),
        ("cuda", "cpu",  "GPU — offload=cpu"),
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
        ("cuda", "gpu",  "GPU — offload=gpu"),
        ("cuda", "cpu",  "GPU — offload=cpu"),
        ("cuda", "disk", "GPU — offload=disk"),
    ]
    x_labels_c = [c[2] for c in gpu_only_configs]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "n_test=200  (GPU vs CPU, n_train=4000)",
            "n_test=10 000  (GPU only, n_train=4000)",
        ],
        horizontal_spacing=0.12,
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
            marker_color=COLOR_CACHE_OFF,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>predict=%{y:.3g} s<extra></extra>",
        ),
        row=1, col=1,
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
            marker_color="#2563eb",
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>RAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1, col=1,
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
            marker_color="#dc2626",
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>VRAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Right panel: predict time + memory (3 traces, GPU only) ---
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
            marker_color=COLOR_CACHE_OFF,
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>predict=%{y:.3g} s<extra></extra>",
        ),
        row=1, col=2,
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
            marker_color="#2563eb",
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>RAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1, col=2,
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
            marker_color="#dc2626",
            marker_line=dict(width=1, color="#1f2937"),
            hovertemplate="%{x}<br>VRAM=%{y:.0f} MB<extra></extra>",
        ),
        row=1, col=2,
    )

    for c in (1, 2):
        fig.update_xaxes(title_text="Device × offload mode", row=1, col=c)
        fig.update_yaxes(type="log", row=1, col=c)
    fig.update_yaxes(title_text="Predict time (s) / Memory (MB)", row=1, col=1)

    _finalize(
        fig,
        title="Figure 3 — Offload comparison: GPU vs CPU (left) and large n_test (right)",
        subtitle="Classification, 10 classes, n_estimators=4, kv_cache=False, n_features=200, n_train=4000. "
                 "Left: n_test=200, GPU vs CPU (offload no-op on CPU). "
                 "Right: n_test=10 000, GPU only — large output tensors make offload modes save ~2.5 GB VRAM. "
                 "Bars: predict time (gray), RAM (blue), VRAM (crimson). Log y.",
        height=560,
        legend_title="Metric",
    )
    return fig


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _finalize(fig: go.Figure, title: str, subtitle: str, height: int, legend_title: str) -> None:
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:#6b7280'>{subtitle}</span>",
            font_size=18,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(
            title=dict(text=f"<b>{legend_title}</b>", font_size=13),
            orientation="h",
            y=-0.22,
            x=0,
            font_size=12,
        ),
        margin=dict(l=70, r=70, t=160, b=160),
        height=height,
    )
    # Lift subplot titles slightly above their default position so they
    # breathe relative to the plot below. Using a relative offset (not a
    # fixed paper-y) keeps multi-row layouts correct: each title stays
    # above its own subplot instead of stacking at the top of the figure.
    fig.for_each_annotation(
        lambda a: a.update(
            y=a.y + 0.03,
            yanchor="bottom",
            font=dict(size=14, color="#1f2937"),
        )
    )
    fig.update_yaxes(
        zeroline=False,
        gridcolor="rgba(0,0,0,0.10)",
        title_font=dict(size=13, color="#1f2937"),
        tickfont=dict(size=11, color="#374151"),
    )
    fig.update_xaxes(
        zeroline=False,
        gridcolor="rgba(0,0,0,0.10)",
        title_font=dict(size=13, color="#1f2937"),
        tickfont=dict(size=11, color="#374151"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/results.csv", help="consolidated results CSV")
    ap.add_argument("--out", default="display/gallery_figures", help="output dir for HTML")
    args = ap.parse_args()

    df = load_results(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builders = {
        "fig1_kv_cache": build_fig1_kv_cache,
        "fig2_memory": build_fig2_vram,
        "fig3_offload": build_fig3_offload,
    }
    for name, fn in builders.items():
        fig = fn(df)
        path = out / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

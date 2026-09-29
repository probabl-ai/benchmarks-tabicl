"""Shared base class for the TabICL classifier and regressor solvers.

Both solvers share the entire inference+measurement pipeline; only the
estimator class, the task they accept, and whether they produce
probabilities differ. Those specifics are declared via class attributes on
the concrete subclass, and everything else lives here.

Measurement windows
-------------------
``run`` always does ``fit`` then ``predict`` inline (benchopt's ``time``
covers both). Peak RAM and peak VRAM are recorded **separately for the fit
and predict phases**: a ``ResourceTracker`` is started before fit and
stopped right after fit (``resources_fit``), then a fresh tracker is started
before predict and stopped after predict (``resources_predict``). A combined
``ram_peak_mb`` / ``vram_peak_mb`` (max of the two windows) is also reported
as the headline aggregate peak.

| kv_cache | benchopt ``time`` | fit-window peak | predict-window peak |
|----------|-------------------|-----------------|---------------------|
| True     | fit + predict     | fit(cache)      | predict             |
| False    | fit + predict     | fit             | predict             |

``fit_time`` and ``predict_time`` are always reported separately as objective
metrics. The per-phase peaks (``ram_peak_fit_mb``, ``vram_peak_predict_mb``,
``vram_peak_fit_mb``, ``vram_peak_predict_mb``) are also reported.
"""

import shutil
import tempfile
import time
from pathlib import Path

import tabicl
from benchopt import BaseSolver

from benchmark_utils.defaults import (
    DEVICE_GRID,
    check_default_batch_size,
    is_device_available,
)
from benchmark_utils.memory_tracking import (
    ResourceTracker,
    cleanup_before_run,
    get_gpu_info,
)

# Verified once at import; the same default applies to both estimators.
_DEFAULT_BATCH_SIZE = check_default_batch_size(tabicl.TabICLClassifier)


class BaseTabICLSolver(BaseSolver):
    """Common TabICL solver logic; subclass to pin an estimator + task.

    Subclass attributes
    -------------------
    name : str
        Benchopt display/CLI identifier.
    estimator_cls : type
        ``tabicl.TabICLClassifier`` or ``tabicl.TabICLRegressor``.
    task : str
        ``"classification"`` or ``"regression"`` — the only task this solver
        runs; other tasks are skipped.
    needs_proba : bool
        If True, ``run`` also calls ``predict_proba`` and returns ``y_score``
        (classifier only).
    test_config : dict
        Per-subclass fast config for ``benchopt test`` (must pin ``task`` and
        ``device``).
    """

    # --- subclass-overridden hooks ----------------------------------------

    # Grid of TabICL inference knobs that affect walltime / memory.
    parameters = {
        "n_estimators": [1, 2, 4, 8],
        "batch_size": [_DEFAULT_BATCH_SIZE],  # default; extend to sweep later
        "kv_cache": [False, True],
        "offload_mode": ["auto", "gpu", "cpu", "disk"],
        # No disk_offload_dir here — it comes from the objective's scratch_dir
        # parameter (see get_objective), so it's set once for all solvers.
        "n_jobs": [-1],  # use all cores on CPU
        "device": DEVICE_GRID,  # explicit; unavailable devices are skipped
        "use_amp": ["auto"],  # automatic mixed precision; extend to sweep later
    }

    # --- shared implementation --------------------------------------------

    def skip(self, X_train, y_train, X_test, task, n_classes, scratch_dir):
        if task != self.task:
            return True, f"{self.name} only handles {self.task}."
        # ``n_classes`` is only meaningful for classification; the cartesian
        # product creates one regression instance per n_classes value with
        # identical data. Keep n_classes=10 as the canonical regression
        # instance and skip the redundant duplicates.
        if task == "regression" and n_classes != 10:
            return True, "n_classes is only used for classification."
        # kv_cache only supports up to 10 classes; skip when above the limit.
        if self.kv_cache and task == "classification" and n_classes > 10:
            return True, "kv_cache=True only supports n_classes <= 10."
        if not is_device_available(self.device):
            return True, f"device {self.device!r} is not available on this host."
        # disk offload needs a scratch dir; skip (not error) so the full grid
        # runs cleanly without crashing on this config.
        if self.offload_mode == "disk" and scratch_dir is None:
            return True, (
                "scratch_dir must be set when offload_mode='disk'; pass it "
                "via the objective, e.g. "
                '-o "TabICL inference[scratch_dir=/scratch/tabicl]".'
            )
        return False, None

    def _resolve_disk_offload_dir(self):
        """Resolve the disk-offload directory for ``offload_mode='disk'``.

        Uses the objective-provided ``scratch_dir`` as the parent, creating a
        ``disk-offload`` subdirectory inside it (so multiple solvers sharing
        the same scratch dir stay isolated). A fresh, unique subdirectory is
        created on every call so a run never reuses offload files left behind
        by a previous run (or by the warm-up estimator). The resolved path is
        stored on ``self._disk_offload_dir_used`` for reporting in results.
        """
        parent = Path(self.scratch_dir) / "disk-offload"
        parent.mkdir(parents=True, exist_ok=True)
        # Unique per-call subdir to isolate this run's offload files.
        run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
        return str(run_dir)

    def _cleanup_disk_offload_dir(self):
        """Delete the per-run disk-offload directory after the run is done.

        The offload files are only needed during fit/predict; once ``run``
        has stored the predictions they can be removed. The path string in
        ``self._disk_offload_dir_used`` is preserved for ``get_result`` to
        report in the results parquet. Errors are ignored (e.g. if the dir
        was already removed or is on a stale NFS handle).
        """
        d = getattr(self, "_disk_offload_dir_used", None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def _build_estimator(self):
        """Construct a fresh estimator with the current parameters.

        ``disk_offload_dir`` is resolved from ``scratch_dir`` whenever a
        scratch dir is available, regardless of ``offload_mode``. This lets
        ``offload_mode='auto'`` fall back to disk offloading (per TabICL's
        API) when VRAM is insufficient. When ``scratch_dir`` is None,
        ``disk_offload_dir`` stays None and ``auto`` simply won't use disk.
        """
        disk_offload_dir = None
        if self.scratch_dir is not None:
            disk_offload_dir = self._resolve_disk_offload_dir()
        self._disk_offload_dir_used = disk_offload_dir
        return self.estimator_cls(
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            kv_cache=self.kv_cache,
            offload_mode=self.offload_mode,
            disk_offload_dir=disk_offload_dir,
            n_jobs=self.n_jobs,
            device=self.device,
            use_amp=self.use_amp,
        )

    def set_objective(self, X_train, y_train, X_test, task, n_classes, scratch_dir):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        # Shared scratch dir from the objective; used by _resolve_disk_offload_dir.
        self.scratch_dir = scratch_dir
        # Default; overwritten when an estimator is built.
        self._disk_offload_dir_used = None
        # Estimator is built in run().
        self.estimator = None
        # Resource trackers are per-phase: _tracker_fit covers the fit window,
        # _tracker_predict covers the predict window. Each is started/stopped
        # independently so peak RAM/VRAM is recorded separately per phase.
        self._tracker_fit = None
        self._tracker_predict = None

    def _predict(self, X):
        """Run the timed prediction on ``X``.

        Returns ``(y_pred, y_score, y_encoder)``:
        * classifier: ``y_pred`` is ``None`` (derived later, untimed, in the
          objective from ``y_score`` + ``y_encoder``), ``y_score`` is the output
          of a single ``predict_proba`` call, and ``y_encoder`` is the fitted
          label encoder. Only one forward pass is run — ``predict`` would just
          redo ``predict_proba`` + argmax + inverse_transform.
        * regressor: ``y_pred`` is the output of ``predict``, ``y_score`` and
          ``y_encoder`` are ``None``.
        """
        if self.needs_proba:
            y_score = self.estimator.predict_proba(X)
            return None, y_score, self.estimator.y_encoder_
        y_pred = self.estimator.predict(X)
        return y_pred, None, None

    def warm_up(self):
        """Untimed fit + predict on a tiny dummy slice to trigger CUDA kernel /
        cuDNN / FA3 JIT autotuning. ``run`` builds its own fresh estimator, so
        this one is discarded.
        """
        import gc

        import numpy as np

        est = self._build_estimator()
        n_feat = self.X_train.shape[1]
        rng = np.random.default_rng(0)
        X_fit = rng.standard_normal((2, n_feat))
        X_pred = rng.standard_normal((2, n_feat))
        if self.task == "classification":
            y_fit = np.zeros(2, dtype=self.y_train.dtype)
        else:
            y_fit = np.zeros(2, dtype=self.y_train.dtype)
        est.fit(X_fit, y_fit)
        if self.needs_proba:
            est.predict_proba(X_pred)
        else:
            est.predict(X_pred)
        del est
        gc.collect()
        cleanup_before_run(self.device)

    # --- timed run --------------------------------------------------------

    def run(self, stop_val):
        # Clean baseline before any allocation: release leftover state from
        # prior runs (gc + empty_cache on the matching accelerator backend).
        cleanup_before_run(self.device)

        self.estimator = self._build_estimator()

        # Fit window: tracker covers the fit (incl. kv_cache build) only.
        self._tracker_fit = ResourceTracker()
        self._tracker_fit.start()
        t0 = time.perf_counter()
        self.estimator.fit(self.X_train, self.y_train)
        t1 = time.perf_counter()
        self.resources_fit = self._tracker_fit.stop()

        # Predict window: fresh tracker, baseline is the post-fit state so
        # the peak reflects only the incremental predict allocations.
        self._tracker_predict = ResourceTracker()
        self._tracker_predict.start()
        self.y_pred, self.y_score, self.y_encoder = self._predict(self.X_test)
        t2 = time.perf_counter()
        self.resources_predict = self._tracker_predict.stop()

        self.fit_time = t1 - t0
        self.predict_time = t2 - t1

        # Predict-phase footprint accounting for the resident fit state (e.g.
        # the kv cache, the fitted estimator) that persists into predict: the
        # predict window's baseline is the post-fit state, so its peak is only
        # the *incremental* predict allocations. Adding the fit window's
        # end-of-window usage (the retained fit state) gives the absolute
        # footprint during predict — closer to a real inference workload.
        ram_predict_abs = (
            self.resources_fit["ram_end_mb"]
            + self.resources_predict["ram_peak_mb"]
        )
        vram_predict_abs = (
            self.resources_fit["vram_end_mb"]
            + self.resources_predict["vram_peak_mb"]
        )

        # Combined peak (max of the fit and predict windows). The predict
        # contribution is the absolute footprint (resident fit state + predict
        # peak), not the pure incremental peak. This is the headline aggregate
        # the figures plot.
        self.resources = {
            "ram_peak_mb": max(self.resources_fit["ram_peak_mb"], ram_predict_abs),
            "vram_peak_mb": max(self.resources_fit["vram_peak_mb"], vram_predict_abs),
            "ram_peak_predict_mb": ram_predict_abs,
            "vram_peak_predict_mb": vram_predict_abs,
        }

        # Offload files are no longer needed once predict is done; remove the
        # per-run temp dir so the scratch dir doesn't fill up across runs.
        # The path string is preserved in _disk_offload_dir_used for get_result.
        self._cleanup_disk_offload_dir()

    def get_result(self):
        # Resolve GPU info for the device the solver actually used (not just
        # what's available on the host), so a CPU run on a GPU machine records
        # device='cpu', gpu_name=None.
        gpu_name, device = get_gpu_info(self.device)
        return dict(
            y_pred=self.y_pred,
            y_score=self.y_score,
            y_encoder=self.y_encoder,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            # Combined peak (max of fit & predict windows) — headline aggregate.
            ram_peak_mb=self.resources["ram_peak_mb"],
            vram_peak_mb=self.resources["vram_peak_mb"],
            # Fit-phase peak: incremental peak within the fit window.
            ram_peak_fit_mb=self.resources_fit["ram_peak_mb"],
            vram_peak_fit_mb=self.resources_fit["vram_peak_mb"],
            # Predict-phase peak: absolute footprint during predict — the fit
            # window's retained state (e.g. kv cache) + the predict window's
            # incremental peak. Closer to a real inference workload.
            ram_peak_predict_mb=self.resources["ram_peak_predict_mb"],
            vram_peak_predict_mb=self.resources["vram_peak_predict_mb"],
            disk_offload_dir=self._disk_offload_dir_used,
            device=device,
            gpu_name=gpu_name,
        )

"""Shared base class for the TabICL classifier and regressor solvers.

Both solvers share the entire inference+measurement pipeline; only the
estimator class, the task they accept, and whether they produce
probabilities differ. Those specifics are declared via class attributes on
the concrete subclass, and everything else lives here.

Measurement windows
-------------------
Two axes cross to form four scenarios:

* ``warmup`` (True/False): whether ``fit`` + a dummy 2-row ``predict`` run
  ahead of time (in ``warm_up``, untimed by benchopt but covered by the
  ResourceTracker) or inline in ``run`` (timed by benchopt).
* ``kv_cache`` (True/False): whether ``fit`` builds the KV cache.

| warmup | kv_cache | benchopt ``time``        | peak RAM/VRAM window              |
|--------|----------|--------------------------|-----------------------------------|
| True   | True     | predict only             | fit(cache) + dummy predict + pred |
| True   | False    | predict only             | fit + dummy predict + predict     |
| False  | True     | fit + predict            | fit(cache) + predict              |
| False  | False    | fit + predict            | fit + predict                     |

``fit_time`` and ``predict_time`` are always reported separately as objective
metrics, so the breakdown is recoverable regardless of the scenario.
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
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

# Number of rows used for the dummy warm-up predict (just enough to trigger
# CUDA kernel autotuning / cuDNN JIT / FA3 JIT for the forward pass).
_WARMUP_N_ROWS = 2


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
        Per-subclass fast config for ``benchopt test`` (must pin ``task``,
        ``device``, and ``warmup``).
    """

    # --- subclass-overridden hooks ----------------------------------------

    # Grid of TabICL inference knobs that affect walltime / memory.
    parameters = {
        "n_estimators": [1, 2, 4, 8],
        "batch_size": [_DEFAULT_BATCH_SIZE],  # default; extend to sweep later
        "kv_cache": [False, True],
        "warmup": [True, False],
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
        # Estimator is built in warm_up (warmup=True) or in run (warmup=False).
        self.estimator = None
        # Resource tracker lives across warm_up + run when warmup=True, so the
        # peak memory window covers the fit (including kv_cache build).
        self._tracker = None
        # Benchopt's _warm_up has a "run once" guard (_warmup_done) that persists
        # across datasets when the solver instance is reused (sequential runs).
        # Clear it so warm_up re-runs for each new dataset — otherwise the
        # estimator from the previous dataset (or None if set_objective reset
        # it) is used by run(), causing a stale/None estimator error.
        self._warmup_done = None

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

    # --- warmup scenarios (1 & 4) -----------------------------------------

    def _warmup_input(self):
        """Return a tiny dummy test set with subtle noise for the warm-up predict.

        Takes the first ``_WARMUP_N_ROWS`` rows of the real test set and adds
        small per-column Gaussian noise (1e-3 of each column's std, floored
        to avoid zero scale on constant columns). This ensures the warm-up
        rows differ from the real test rows so the timed predict cannot
        benefit from any data-level memoization, while the warm-up still
        triggers forward-pass kernel / cuDNN / FA3 autotuning.
        """
        X = np.array(self.X_test[:_WARMUP_N_ROWS], dtype=float, copy=True)
        # Fixed seed: the noise only needs to differ from the real data, not
        # be random across runs.
        rng = np.random.default_rng(0)
        col_std = X.std(axis=0, keepdims=True)
        # 1e-3 relative scale + 1e-6 floor for constant/zero-variance columns.
        noise_scale = 1e-3 * col_std + 1e-6
        X += rng.standard_normal(X.shape) * noise_scale
        return X

    def warm_up(self):
        """Pre-fit + dummy predict ahead of the timed run (warmup=True only).

        Starts the ResourceTracker here so peak RAM/VRAM covers the fit
        (including the kv_cache build) and the dummy forward pass that warms
        up CUDA kernels / cuDNN JIT / FA3 JIT. The tracker keeps running into
        ``run``, which only does the real predict.
        """
        # Clean baseline before any allocation: release leftover state from
        # prior runs (gc + empty_cache on the matching accelerator backend).
        cleanup_before_run(self.device)

        if not self.warmup:
            return  # scenario 2 or 3: everything happens in run()

        self.estimator = self._build_estimator()

        self._tracker = ResourceTracker()
        self._tracker.start()

        t0 = time.perf_counter()
        self.estimator.fit(self.X_train, self.y_train)
        self.fit_time = time.perf_counter() - t0

        # Dummy predict on a tiny noisy slice to trigger forward-pass kernel
        # autotuning without the full predict cost (and without reusing the
        # exact rows the timed predict will see — see ``_warmup_input``).
        self._predict(self._warmup_input())

    # --- timed run --------------------------------------------------------

    def run(self, stop_val):
        if self.warmup:
            # Estimator and tracker are already live from warm_up; just do the
            # real (timed) predict.
            t0 = time.perf_counter()
            self.y_pred, self.y_score, self.y_encoder = self._predict(self.X_test)
            self.predict_time = time.perf_counter() - t0
            self.resources = self._tracker.stop()
        else:
            self.estimator = self._build_estimator()

            self._tracker = ResourceTracker()
            self._tracker.start()

            t0 = time.perf_counter()
            self.estimator.fit(self.X_train, self.y_train)
            t1 = time.perf_counter()
            self.y_pred, self.y_score, self.y_encoder = self._predict(self.X_test)
            t2 = time.perf_counter()

            self.resources = self._tracker.stop()
            self.fit_time = t1 - t0
            self.predict_time = t2 - t1

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
            ram_peak_mb=self.resources["ram_peak_mb"],
            vram_peak_mb=self.resources["vram_peak_mb"],
            disk_offload_dir=self._disk_offload_dir_used,
            device=device,
            gpu_name=gpu_name,
        )

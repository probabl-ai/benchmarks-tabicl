"""Shared base class for the TabICL classifier and regressor solvers.

Both solvers share the entire inference+measurement pipeline; only the
estimator class, the task they accept, and whether they produce
probabilities differ. Those specifics are declared via class attributes on
the concrete subclass, and everything else lives here.
"""

import time
from pathlib import Path

import tabicl
from benchopt import BaseSolver

from benchmark_utils.defaults import (
    DEVICE_GRID, check_default_batch_size, is_device_available,
)
from benchmark_utils.memory_tracking import ResourceTracker, cleanup_before_run

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
        a compatible ``device``).
    """

    # --- subclass-overridden hooks ----------------------------------------

    # Grid of TabICL inference knobs that affect walltime / memory.
    parameters = {
        "n_estimators": [1, 2, 4, 8],
        "batch_size": [_DEFAULT_BATCH_SIZE],  # default; extend to sweep later
        "kv_cache": [False, True],
        "offload_mode": ["auto", "gpu", "cpu", "disk"],
        # Path for memory-mapped offload files when offload_mode='disk'.
        # Required when offload_mode='disk' (a ValueError is raised if None).
        "disk_offload_dir": [None],
        "n_jobs": [-1],             # use all cores on CPU
        "device": DEVICE_GRID,       # explicit; unavailable devices are skipped
    }

    # --- shared implementation --------------------------------------------

    def skip(self, X_train, y_train, X_test, task, n_classes):
        if task != self.task:
            return True, f"{self.name} only handles {self.task}."
        # ``n_classes`` is only meaningful for classification; the cartesian
        # product creates one regression instance per n_classes value with
        # identical data. Keep n_classes=10 as the canonical regression
        # instance and skip the redundant duplicates.
        if task == "regression" and n_classes != 10:
            return True, "n_classes is only used for classification."
        if not is_device_available(self.device):
            return True, f"device {self.device!r} is not available on this host."
        return False, None

    def _resolve_disk_offload_dir(self):
        """Resolve the disk-offload directory for ``offload_mode='disk'``.

        The user must provide ``disk_offload_dir`` when using
        ``offload_mode='disk'``; raises ``ValueError`` if it is missing. The
        directory is created if needed and the resolved path is stored on
        ``self._disk_offload_dir_used`` so it can be reported in the results.
        """
        d = self.disk_offload_dir
        if d is None:
            raise ValueError(
                "disk_offload_dir must be set when offload_mode='disk'. "
                "Pass it via the CLI, e.g. "
                f"-s \"{self.name}[offload_mode=disk,"
                "disk_offload_dir=/scratch/tabicl]\"."
            )
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _build_estimator(self):
        """Construct a fresh estimator with the current parameters.

        Recreated at the start of ``run`` (not just ``set_objective``) so the
        timed measurement starts from a clean state — important for
        ``kv_cache=True``, where ``fit`` builds the cache: without this, the
        cache warmed up in ``warm_up`` would carry over and the timed ``fit``
        would measure a refill rather than a clean build.
        """
        disk_offload_dir = None
        if self.offload_mode == "disk":
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
        )

    def set_objective(self, X_train, y_train, X_test, task, n_classes):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        # Default; overwritten in `run` / `warm_up` when offload_mode='disk'.
        self._disk_offload_dir_used = None
        # Build once for warm_up; ``run`` rebuilds a fresh estimator so the
        # timed measurement starts clean (see ``_build_estimator``).
        self.estimator = self._build_estimator()

    def warm_up(self):
        # Absorb one-time costs (checkpoint download, CUDA kernel autotuning,
        # cuDNN JIT, FA3 JIT) out of the measured run. The downloaded
        # checkpoint is cached on disk for subsequent runs. For
        # ``kv_cache=True`` this also fills the cache once — but the timed
        # ``run`` rebuilds a fresh estimator, so it measures a clean cache
        # build, not a refill.
        self.run_once()

    def _predict(self):
        """Task-specific prediction; override in the subclass if needed.

        Returns ``(y_pred, y_score)`` where ``y_score`` is ``None`` when the
        estimator has no ``predict_proba``.
        """
        y_pred = self.estimator.predict(self.X_test)
        y_score = None
        if self.needs_proba:
            y_score = self.estimator.predict_proba(self.X_test)
        return y_pred, y_score

    def run(self, stop_val):
        # Release the warm-up estimator and reclaim its memory so the
        # ResourceTracker baseline is clean (otherwise the warm-up model's
        # RSS / VRAM inflates the baseline and corrupts the incremental peak).
        self.estimator = None
        cleanup_before_run()
        # Fresh estimator: the timed fit/predict start from a clean state.
        self.estimator = self._build_estimator()

        tracker = ResourceTracker()
        tracker.start()

        t0 = time.perf_counter()
        self.estimator.fit(self.X_train, self.y_train)
        t1 = time.perf_counter()
        self.y_pred, self.y_score = self._predict()
        t2 = time.perf_counter()

        self.resources = tracker.stop()
        self.fit_time = t1 - t0
        self.predict_time = t2 - t1

    def get_result(self):
        return dict(
            y_pred=self.y_pred,
            y_score=self.y_score,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            ram_peak_mb=self.resources["ram_peak_mb"],
            vram_peak_mb=self.resources["vram_peak_mb"],
            disk_offload_dir=self._disk_offload_dir_used,
        )

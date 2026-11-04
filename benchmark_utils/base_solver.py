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

Batch observations
------------------
The estimator runs with ``verbose=True`` and the fit and predict logs are
captured to count how many batches each transformer stage processed. The counts
are stored in ``self.observations`` as ``{fit,predict}_nb_batches_{tf_col,
tf_row,tf_icl}`` and reported as objective metrics. A stage that does not run
(every stage of the fit when ``kv_cache=False``; ``tf_icl`` of the fit when
``kv_cache='repr'``, which caches row representations instead) stays ``None``.

Simulated memory budget
-----------------------
``run`` starts by cleaning the device, then -- on an accelerator -- reads the
driver-reported free VRAM and records it as ``vram_available_mb``. That is the
budget the run really starts from, before any checkpoint is loaded.

The ``available_memory`` parameter (MB) turns this into a controlled
experiment: when set, the solver allocates a random float32 tensor sized to
leave only ``available_memory`` MB free on the device, then runs the benchmark
as usual. This emulates a smaller GPU on a large one, which is what makes the
estimator's auto-batching policy (and the time it costs) observable without
touching the hardware. It is CUDA-only: combinations with any other device are
skipped. The reserve is dropped at the start of the next ``run``, so it never
contaminates the next cell's baseline. Setting it above the measured
free memory is an error, since no allocation could honour it.

``vram_available_mb`` is recorded in the results but is not part of the
consolidated CSV.
"""

import contextlib
import gc
import io
import re
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
    allocate_vram_reserve,
    cleanup_before_run,
    get_available_vram_mb,
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
        "kv_cache": [False, "repr", True],
        # Offloading comparisons are not part of the gallery study: keep the
        # estimator default as the only value so the results table records it.
        "offload_mode": ["auto"],
        # Simulated memory budget, in MB. ``None`` (default) runs on the
        # device as-is. A value below the free VRAM measured at the start of
        # ``run`` is enforced by allocating a random-float reserve until only
        # that much is left free, which emulates a smaller GPU and makes the
        # auto-batching policy observable. CUDA only: combinations with any
        # other device are skipped. See ``_reserve_vram``.
        "available_memory": [None],
        # No disk_offload_dir here — it comes from the objective's scratch_dir
        # parameter (see get_objective), so it's set once for all solvers.
        # TabICL's default: None leaves PyTorch's thread count alone (the
        # number of physical CPU cores). ``-1`` would use every logical core.
        "n_jobs": [None],
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
            return True, f"kv_cache={self.kv_cache!r} only supports n_classes <= 10."
        if not is_device_available(self.device):
            return True, f"device {self.device!r} is not available on this host."
        # The simulated memory budget works by occupying driver-visible VRAM
        # until only ``available_memory`` MB are left free. That needs a CUDA
        # device: it is the only backend the benchmark queries for free
        # memory, and the only one titrated this way.
        if self.available_memory is not None and self.device != "cuda":
            return True, (
                f"available_memory={self.available_memory} MB is only supported "
                f"with device='cuda'; got device={self.device!r}. "
                "Set available_memory=None to run on this device."
            )
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

    def _reserve_vram(self):
        """Leave ``available_memory`` MB free on the device.

        Called at the start of ``run``, right after the cleanup and the free
        memory measurement, so the reserve is part of the baseline rather than
        of the measured peak. ``None`` is a no-op; a value above the measured
        free memory is an error, because it would describe a device with more
        memory than this one has.

        The reserve is a single tensor held in ``self._vram_reserve`` and
        dropped by ``_cleanup_before_run`` at the end of the run that allocated
        it.
        """
        self._vram_reserve = None
        if self.available_memory is None:
            return
        if self.available_memory < 0:
            raise ValueError(
                f"available_memory must be non-negative, got {self.available_memory}."
            )
        measured = self.vram_available_mb
        if measured is None:
            raise RuntimeError(
                f"available_memory={self.available_memory} MB was requested, but the "
                f"free memory of device {self.device!r} cannot be queried."
            )
        if self.available_memory > measured:
            raise RuntimeError(
                f"available_memory={self.available_memory} MB exceeds the "
                f"{measured:.0f} MB free on {self.device!r} after cleanup: cannot "
                "emulate a device with more memory than this one has."
            )
        self._vram_reserve = allocate_vram_reserve(self.device, self.available_memory)

    def _cleanup_before_run(self):
        """Release the previous cell's state, then clean the device.

        Drops the fitted estimator (with its KV cache, which lives on the
        device) and the simulated reserve tensor, then calls the module-level
        ``cleanup_before_run`` (gc + synchronize + ``empty_cache``). It runs at
        the start and in the ``finally`` of every ``run``, and at the end of
        ``warm_up``: benchopt keeps every solver instance alive for the whole
        invocation, so whatever a cell still references stays charged to the
        driver and would shrink the budget measured by the next one.
        """
        self.estimator = None
        self._vram_reserve = None
        # Drop the references first, let the collector reclaim them, and only
        # then empty the cache: a live tensor cannot be handed back to the
        # driver, and a dead one only becomes reclaimable after a collection.
        gc.collect()
        cleanup_before_run(self.device)

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
            verbose=True,
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
        # Free VRAM measured at the start of run (None on CPU); recorded as a
        # result but not consolidated. Simulated reserve (a single tensor, or
        # None), dropped at the end of every run.
        self.vram_available_mb = None
        self._vram_reserve = None
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
        """Untimed fit + predict on a tiny dummy slice.

        Purpose is to trigger CUDA kernel / cuDNN / FA3 JIT autotuning. ``run``
        builds its own fresh estimator, so this one is discarded.
        """
        import numpy as np
        from sklearn.base import clone

        # Untimed and not benchmarked: keep the estimator quiet.
        est = clone(self._build_estimator()).set_params(verbose=False)
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
        self._cleanup_before_run()

    # --- batch observations ------------------------------------------------

    def _record_nb_batches(self, phase, output):
        """Record the batches processed by each transformer stage in a phase.

        TabICL's ``InferenceManager`` prints one "estimated batch size" line
        per stage it runs, and a tqdm "Processing <stage>: i/n" bar only when
        that stage is split into several batches. The line is therefore always
        there for a stage that ran, with two exceptions, both about ``tf_icl``:

        * when building a ``repr`` cache (see ``forward_with_cache``), which
          bakes ``y_train`` into the row representations instead of running the
          ICL transformer;
        * in hierarchical classification (any class count above the model's
          10-class head), where the ICL head is applied once per tree node with
          ``auto_batch=False`` (see ``_predict_hierarchical``) and the manager
          returns before printing. There is no single batch count to record, so
          the value stays ``None``.

        On CPU there is nothing to parse: the manager returns before the
        verbose line is printed, because auto-batching does not exist there
        (``if self.exe_device.type == "cpu": return self._run_forward(...)``).
        Every stage that runs does so in a single pass, so the count is 1.
        """
        # With kv_cache=False, fit runs no transformer at all: nothing to parse,
        # every fit count stays None.
        if phase == "fit" and not self.kv_cache:
            return
        # No batching on CPU: record the single pass instead of looking for a
        # log line that TabICL never prints there.
        on_cpu = self.device == "cpu"
        prefix = f"{phase}_nb_batches_"
        not_run = {"tf_icl"} if (phase == "fit" and self.kv_cache == "repr") else set()
        for key in self.observations:
            if not key.startswith(prefix):
                continue
            stage = key.removeprefix(prefix)
            if stage in not_run:
                continue
            if on_cpu:
                self.observations[key] = 1
                continue
            if f"estimated batch size for {stage}:" not in output:
                # Hierarchical classification runs the ICL head once per tree
                # node without auto-batching, so TabICL never prints the line.
                # Leave the observation unset rather than invent a count.
                if stage == "tf_icl" and phase == "predict":
                    continue
                raise RuntimeError(
                    f"No batching log for {stage} during {phase}; cannot record {key}."
                )
            totals = re.findall(rf"Processing {stage}:.*?(\d+)/(\d+) \[", output)
            self.observations[key] = int(totals[-1][1]) if totals else 1

    # --- timed run --------------------------------------------------------

    def run(self, stop_val):
        # Timed entry point. The cleanup is in a ``finally`` so that a cell
        # that aborts (an OOM, typically) cannot leave its reserve or its
        # fitted estimator charged to the device: benchopt keeps every solver
        # object alive for the whole invocation, so nothing else would release
        # them before the end of the run.
        try:
            self._run(stop_val)
        finally:
            self._cleanup_before_run()

    def _run(self, stop_val):
        # Release whatever the previous repetition left behind (estimator, KV
        # cache, simulated reserve) and clean the device, so the free-VRAM
        # baseline measured below is not polluted by earlier cells.
        self._cleanup_before_run()

        # Memory budget this run actually starts from: the driver's free VRAM
        # right after the cleanup, before any checkpoint is loaded. Recorded
        # for every accelerator run, with or without a simulated budget.
        self.vram_available_mb = (
            None if self.device == "cpu" else get_available_vram_mb(self.device)
        )
        # Emulate a smaller device when ``available_memory`` is set. Must run
        # before the estimator is built so the checkpoint is loaded out of the
        # simulated budget, exactly as on the smaller GPU.
        self._reserve_vram()

        self.estimator = self._build_estimator()

        # Batches processed by each transformer stage, over fit and predict.
        # Stages that do not run keep their None default.
        self.observations = {
            f"{phase}_nb_batches_{stage}": None
            for phase in ("fit", "predict")
            for stage in ("tf_col", "tf_row", "tf_icl")
        }

        # Fit window: tracker covers the fit (incl. kv_cache build) only.
        self._tracker_fit = ResourceTracker()
        self._tracker_fit.start()
        # Capture TabICL's verbose stage logs (tqdm bars go to stderr) to count
        # the batches processed by each transformer stage.
        stdout, stderr = io.StringIO(), io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.estimator.fit(self.X_train, self.y_train)
        t1 = time.perf_counter()
        self.resources_fit = self._tracker_fit.stop()
        self._record_nb_batches("fit", stdout.getvalue() + stderr.getvalue())

        # Predict window: fresh tracker, baseline is the post-fit state so
        # the peak reflects only the incremental predict allocations.
        self._tracker_predict = ResourceTracker()
        self._tracker_predict.start()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.y_pred, self.y_score, self.y_encoder = self._predict(self.X_test)
        t2 = time.perf_counter()
        self.resources_predict = self._tracker_predict.stop()
        self._record_nb_batches("predict", stdout.getvalue() + stderr.getvalue())

        self.fit_time = t1 - t0
        self.predict_time = t2 - t1

        # Predict-phase footprint accounting for the resident fit state (e.g.
        # the kv cache, the fitted estimator) that persists into predict: the
        # predict window's baseline is the post-fit state, so its peak is only
        # the *incremental* predict allocations. Adding the fit window's
        # end-of-window usage (the retained fit state) gives the absolute
        # footprint during predict — closer to a real inference workload.
        ram_predict_abs = (
            self.resources_fit["ram_end_mb"] + self.resources_predict["ram_peak_mb"]
        )
        vram_predict_abs = (
            self.resources_fit["vram_end_mb"] + self.resources_predict["vram_peak_mb"]
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
            # Driver-reported free VRAM at the start of run, before loading
            # anything (None on CPU). Simulation target is the solver param
            # ``available_memory``, recorded by benchopt as p_solver_*.
            vram_available_mb=self.vram_available_mb,
            **self.observations,
        )

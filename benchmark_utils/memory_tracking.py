"""Peak RAM and VRAM tracking helpers for the TabICL benchmark.

Both metrics are reported as **incremental** peaks: the value at the moment
``start`` is called is recorded as a baseline and subtracted from the peak
observed during the measured work. This isolates the memory cost of the timed
``fit`` + ``predict`` from the constant process footprint (interpreter,
loaded libraries, the already-downloaded checkpoint, etc.).

* **RAM**: sampled from ``psutil.Process().memory_info().rss`` on a
  background thread (``resource.getrusage`` only reports a process-wide,
  monotonic peak that cannot be reset between solvers). Reported as
  ``peak_rss - baseline_rss``.
* **VRAM**: read through ``torch.cuda.max_memory_allocated`` when a CUDA
  device is available. Reported as
  ``max_memory_allocated - baseline_allocated`` (the baseline is
  ``torch.cuda.memory_allocated()`` at ``start``, and the peak counter is
  reset right after so it only tracks the measured window).

A ``ResourceTracker`` is used twice per solver run: once around ``fit``
(capturing the fit-window peak, including the kv_cache build) and once
around ``predict`` (capturing the predict-window peak). The two windows
are independent: ``stop`` returns the incremental peak for its window and
``start`` resets the baseline, so the fit and predict peaks are reported
separately.

``cleanup_before_run(device)`` should be called *before*
``ResourceTracker.start`` to release any leftover state from prior runs:
always ``gc.collect()``, plus ``synchronize`` + ``empty_cache`` on the
matching accelerator backend (cuda/mps/xpu) when available. CPU has no
torch-level allocator cache to clear, so only the gc runs.

``torch`` is imported defensively: the benchmark runs on CPU-only machines
too, and the `Objective` (which calls ``get_gpu_info``) must stay importable
without it.
"""

import contextlib
import threading
import time

import psutil

try:  # optional: only present when a solver pulls in tabicl/torch
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - CPU-only / torch-free envs
    torch = None
    _HAS_TORCH = False


class ResourceTracker:
    """Measure peak RAM (MB) and peak VRAM (MB) between ``start`` and ``stop``.

    Usage::

        tracker = ResourceTracker()
        tracker.start()
        ...  # the work to measure (e.g. fit + predict)
        peak = tracker.stop()   # -> {"ram_peak_mb": ..., "vram_peak_mb": ...}
    """

    def __init__(self, interval=0.05):
        self.interval = interval
        self.proc = psutil.Process()
        self.baseline_rss = 0
        self.peak_rss = 0
        self._stop_event = None
        self._thread = None

    def start(self):
        """Reset baselines and begin background RSS sampling.

        For RAM, ``baseline_rss`` is the resident set size at the moment
        ``start`` is called; the reported peak is ``peak_rss - baseline_rss``
        (the incremental memory used by the measured work, not the total
        process footprint). For VRAM, ``vram_baseline`` is the currently
        allocated CUDA memory at ``start``; the reported peak is
        ``max_memory_allocated - vram_baseline`` (same incremental semantics).

        Call ``start`` *after* releasing any leftover state from a warm-up
        run (delete the old estimator, ``gc.collect()``,
        ``torch.cuda.empty_cache()``) so the baseline is clean.
        """
        self.baseline_rss = self.proc.memory_info().rss
        self.peak_rss = self.baseline_rss
        self.vram_baseline = 0
        self._stop_event = threading.Event()

        def _sample():
            while not self._stop_event.is_set():
                try:
                    rss = self.proc.memory_info().rss
                    if rss > self.peak_rss:
                        self.peak_rss = rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()

        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                self.vram_baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover - defensive
                pass

    def stop(self):
        """Stop sampling and return the incremental usage dict.

        Returns, for both RAM and VRAM, two quantities (all in MB, incremental
        relative to the baseline captured at ``start``):

        * ``{ram,vram}_peak_mb``: the peak usage observed during the window.
        * ``{ram,vram}_end_mb``: the usage at the end of the window — i.e. the
          memory *retained* by the work (fitted state, kv cache, etc.) that
          persists past the window. Callers can add the prior window's
          ``_end_mb`` to the current window's ``_peak_mb`` to get the absolute
          footprint during the current window.
        """
        if self._stop_event is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        # Catch a final RSS reading after the thread stopped; this doubles as
        # the end-of-window RSS for the ram_end_mb field below.
        final_rss = self.peak_rss
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            final_rss = self.proc.memory_info().rss
            self.peak_rss = max(self.peak_rss, final_rss)

        vram_peak_mb = 0.0
        vram_end_mb = 0.0
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                vram_peak_mb = max(0.0, peak - self.vram_baseline) / (1024**2)
                vram_end_mb = max(
                    0.0, torch.cuda.memory_allocated() - self.vram_baseline
                ) / (1024**2)
            except Exception:  # pragma: no cover - defensive
                vram_peak_mb = 0.0
                vram_end_mb = 0.0

        return {
            "ram_peak_mb": (self.peak_rss - self.baseline_rss) / (1024**2),
            "ram_end_mb": max(0.0, final_rss - self.baseline_rss) / (1024**2),
            "vram_peak_mb": float(vram_peak_mb),
            "vram_end_mb": float(vram_end_mb),
        }


def cleanup_before_run(device="cpu"):
    """Release leftover state so the memory baseline is clean before tracking.

    Always forces Python garbage collection (reclaims dead tensor refs on
    any device). For accelerator backends (cuda/mps/xpu) also synchronizes
    the device and empties its caching allocator back to the driver — the
    same API exists on ``torch.cuda``, ``torch.mps`` and ``torch.xpu``.

    Parameters
    ----------
    device : str, default "cpu"
        The device the solver will run on. "cpu" only does ``gc.collect()``
        (there is no torch-level CPU allocator cache to clear).
    """
    import gc

    gc.collect()
    if not _HAS_TORCH or device == "cpu":
        return
    backend = getattr(torch, device, None)
    if backend is None or not hasattr(backend, "is_available"):
        return  # backend module not present in this torch build
    try:
        if not backend.is_available():
            return
        backend.synchronize()
        backend.empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass


def get_available_vram_mb(device="cpu"):
    """Return the driver-reported free VRAM (MB) for ``device``.

    Unlike ``torch.cuda.memory_allocated`` (which only sees this process's
    tensors), the backend's ``mem_get_info`` reports what the *driver* still
    has free, i.e. it also accounts for other processes on the same GPU. That
    is the quantity the auto-batcher works from, so it is the one worth
    recording.

    This is the same query TabICL's ``InferenceManager.get_available_gpu_memory``
    performs on CUDA, in the same unit (MB). That method additionally
    synchronizes the device and calls ``empty_cache()`` first; the solver here
    does that beforehand through ``cleanup_before_run``, so both return the
    same driver reading. Unlike TabICL's, this helper returns ``None`` rather
    than ``0.0`` when the backend cannot report memory, so that "unknown" is
    never mistaken for "no memory left".

    Returns
    -------
    float or None
        Free memory in MB, or ``None`` for ``"cpu"``, when torch is missing,
        or when the backend exposes no memory query (e.g. MPS).
    """
    if not _HAS_TORCH or device == "cpu":
        return None
    backend = getattr(torch, device, None)
    if backend is None:
        return None
    mem_get_info = getattr(backend, "mem_get_info", None)
    if not callable(mem_get_info):
        return None
    try:
        try:
            free, _total = mem_get_info(torch.device(device))
        except TypeError:
            # Some backends expose mem_get_info() without a device argument.
            free, _total = mem_get_info()
    except Exception:  # pragma: no cover - defensive
        return None
    return free / (1024**2)


def allocate_vram_reserve(device, target_free_mb, tolerance_mb=1.0):
    """Occupy VRAM with one tensor, leaving about ``target_free_mb`` free.

    The tensor is sized from the free memory measured right before the
    allocation, so the target is reached without knowing the total device
    capacity. It is filled with random floats (rather than a bare
    ``torch.empty``) so the pages are actually committed, which is what the
    driver's free-memory counter reflects.

    The size is deliberately **conservative**: ``tolerance_mb`` is left on top
    of the target, and the request is rounded down to a 2 MiB multiple, which
    is the granularity the allocator uses for large segments. The reserve
    therefore never overshoots the target, and the few remaining MB of slack
    are irrelevant at the GB-scale accuracy this is used for.

    Parameters
    ----------
    device : str
        Accelerator to allocate on; ``"cpu"`` is rejected.
    target_free_mb : float
        Free memory, in MB, to leave available once the reserve is in place.
    tolerance_mb : float, default 1.0
        Extra free memory left on top of the target, as a safety margin.

    Returns
    -------
    torch.Tensor or None
        The reserve, or ``None`` when the device is already within the
        tolerance of the target. Keep the reference alive to keep the memory
        occupied; drop it and call :func:`cleanup_before_run` to release it.

    Raises
    ------
    RuntimeError
        If the device cannot report its free memory.
    """
    if not _HAS_TORCH or device == "cpu":
        raise ValueError(f"VRAM reservation requires an accelerator, got {device!r}.")

    free_mb = get_available_vram_mb(device)
    if free_mb is None:
        raise RuntimeError(
            f"device {device!r} cannot report its free memory; "
            "cannot simulate a restricted memory budget."
        )

    missing_mb = free_mb - float(target_free_mb) - float(tolerance_mb)
    mib2 = 2 * 1024**2
    n_bytes = int(missing_mb * 1024**2) // mib2 * mib2
    if n_bytes < 10 * 1024**2:
        # Nothing meaningful to reserve: either the device is already at the
        # target, or the reserve would fall in the allocator's 20 MiB bucket.
        return None

    tensor = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    tensor.uniform_()
    return tensor


def get_gpu_info(device="cpu"):
    """Return ``(gpu_name, device_str)`` for the given device.

    Reports the GPU name *for the requested device only*, so a CPU run on a
    machine that also has a GPU correctly returns ``(None, "cpu")`` instead of
    picking up the accelerator.

    Parameters
    ----------
    device : str, default "cpu"
        The device the solver actually used. One of ``"cpu"``, ``"cuda"``,
        ``"mps"``, ``"xpu"``.

    Returns
    -------
    gpu_name : str or None
        The accelerator's model name, or ``None`` for ``"cpu"``.
    device : str
        Echoed back for recording in the results.

    Raises
    ------
    RuntimeError
        If the requested accelerator backend is not available or its name
        cannot be retrieved (e.g. a non-CUDA device on a torch build without
        that backend). The solver's ``skip`` should have caught this
        beforehand, so reaching this error indicates a logic bug.
    """
    if device == "cpu" or not _HAS_TORCH:
        return (None, device if device else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available.")
        return (torch.cuda.get_device_name(0), "cuda")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("mps requested but not available.")
        return (mps.get_name(), "mps")
    if device == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("xpu requested but not available.")
        return (xpu.get_device_name(0), "xpu")
    raise RuntimeError(f"Unknown device {device!r}.")

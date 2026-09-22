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

``cleanup_before_run`` should be called *before* ``ResourceTracker.start`` to
release any leftover state from a warm-up run (delete the old estimator,
``gc.collect()``, ``torch.cuda.empty_cache()``) so the baseline is clean.

``torch`` is imported defensively: the benchmark runs on CPU-only machines
too, and the `Objective` (which calls ``get_gpu_info``) must stay importable
without it.
"""

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
        """Stop sampling and return the incremental peak usage dict."""
        if self._stop_event is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        # Catch a final RSS reading after the thread stopped.
        try:
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        vram_peak_mb = 0.0
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                vram_peak_mb = max(0.0, peak - self.vram_baseline) / (1024 ** 2)
            except Exception:  # pragma: no cover - defensive
                vram_peak_mb = 0.0

        return {
            "ram_peak_mb": (self.peak_rss - self.baseline_rss) / (1024 ** 2),
            "vram_peak_mb": float(vram_peak_mb),
        }


def cleanup_before_run():
    """Release leftover state from a warm-up run so the memory baseline is clean.

    Call this in ``Solver.run`` right before starting the ``ResourceTracker``:
    it deletes references to the previous (warm-up) estimator, forces Python
    garbage collection, and empties the CUDA caching allocator back to the
    GPU driver. Without this the baseline RSS / VRAM is inflated by warm-up
    residue, making the incremental peak measurement inaccurate.
    """
    import gc
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass


def get_gpu_info():
    """Return ``(gpu_name, device_str)`` for the active accelerator.

    Returns ``(None, "cpu")`` when no accelerator is available or torch is
    absent.
    """
    if not _HAS_TORCH:
        return (None, "cpu")
    try:
        if torch.cuda.is_available():
            return (torch.cuda.get_device_name(0), "cuda")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return (xpu.get_device_name(0), "xpu")
        if getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            return ("Apple Silicon GPU", "mps")
    except Exception:  # pragma: no cover - defensive
        pass
    return (None, "cpu")
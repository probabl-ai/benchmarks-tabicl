"""Guards against silent changes of TabICL estimator defaults.

``check_default_batch_size`` hardcodes the batch size the benchmark assumes
(8) and verifies, via ``inspect``, that it still matches the actual default of
the TabICL estimator class. If TabICL ships a new default, importing a solver
fails loudly so the benchmark's ``batch_size`` grid gets updated on purpose,
not by accident.
"""

import inspect

# Devices the benchmark sweeps over. ``device`` must always be explicit
# (never ``None``) so the selected accelerator is unambiguous in the results.
DEVICE_GRID = ["cpu", "cuda", "mps", "xpu"]


def is_device_available(device):
    """Check at runtime whether ``device`` is usable on this machine.

    Uses ``torch`` defensively (the benchmark runs on CPU-only machines too).
    """
    if device == "cpu":
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free env
        return False
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        backends = getattr(torch.backends, "mps", None)
        return bool(backends is not None and backends.is_available())
    if device == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    return False


def check_default_batch_size(estimator_cls, hardcoded=8):
    """Return ``hardcoded`` after checking it matches the estimator default.

    Parameters
    ----------
    estimator_cls : type
        A TabICL estimator class (``TabICLClassifier`` or ``TabICLRegressor``)
        whose ``__init__`` accepts a ``batch_size`` parameter.
    hardcoded : int, default 8
        The batch size this benchmark assumes. Must match the estimator's
        actual default; otherwise a ``RuntimeError`` is raised at import time.

    Returns
    -------
    int
        The verified ``hardcoded`` value, to plug into the solver's
        ``parameters`` grid.
    """
    sig = inspect.signature(estimator_cls.__init__)
    actual = sig.parameters["batch_size"].default
    if actual != hardcoded:
        raise RuntimeError(
            f"Default `batch_size` of {estimator_cls.__name__} changed: "
            f"expected {hardcoded!r}, got {actual!r}. "
            f"Update the `batch_size` grid in the tabicl solvers and the "
            f"`hardcoded` value in benchmark_utils.defaults accordingly."
        )
    return hardcoded

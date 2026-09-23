"""Package-version provenance for the TabICL benchmark.

Records the installed ``torch`` and ``tabicl`` versions so they travel with
each result row. Uses ``importlib.metadata`` (no heavy import; returns
``None`` when the package is absent) so the ``Objective`` stays importable in
torch-free environments (e.g. ``benchopt test --skip-install`` on a bare CI
runner).
"""

import importlib.metadata


def _safe_version(name):
    """Return the installed version of ``name``, or ``None`` if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


#: Resolved once at import. torch/tabl are installed before any solver runs
#: in a real benchmark; ``None`` is the correct value when they are not.
PACKAGE_VERSIONS = {
    "torch_version": _safe_version("torch"),
    "tabicl_version": _safe_version("tabicl"),
}

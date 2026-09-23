import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    Skip/xfail solvers that cannot run on a particular architecture (e.g. a
    CUDA-only build on a CPU CI runner). No-op by default.
    """
    pass

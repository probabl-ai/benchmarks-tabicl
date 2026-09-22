"""TabICL regressor solver.

All shared inference/measurement logic lives in
``benchmark_utils.base_solver.BaseTabICLSolver``; this file only pins the
estimator class, the task, and the fast test config.
"""

import tabicl

from benchmark_utils.base_solver import BaseTabICLSolver


class Solver(BaseTabICLSolver):

    name = "TabICL-Regressor"
    estimator_cls = tabicl.TabICLRegressor
    task = "regression"
    needs_proba = False

    test_config = {
        "dataset": {
            "name": "simulated",
            "n_samples": 200,
            "n_features": 5,
            "n_classes": 10,
            "task": "regression",
        },
        # Shrink the solver grid for `benchopt test` speed.
        "n_estimators": 1,
        "kv_cache": False,
        "offload_mode": "auto",
        "device": "cpu",
    }

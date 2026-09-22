"""TabICL classifier solver.

All shared inference/measurement logic lives in
``benchmark_utils.base_solver.BaseTabICLSolver``; this file only pins the
estimator class, the task, and the fast test config.
"""

import tabicl

from benchmark_utils.base_solver import BaseTabICLSolver


class Solver(BaseTabICLSolver):

    name = "TabICL-Classifier"
    estimator_cls = tabicl.TabICLClassifier
    task = "classification"
    needs_proba = True

    test_config = {
        "dataset": {
            "name": "simulated",
            "n_samples": 200,
            "n_features": 5,
            "n_classes": 2,
            "task": "classification",
        },
        # Shrink the solver grid for `benchopt test` speed.
        "n_estimators": 1,
        "kv_cache": False,
        "offload_mode": "auto",
        "device": "cpu",
    }

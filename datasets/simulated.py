"""Simulated tabular datasets for the TabICL benchmark.

Uses scikit-learn's ``make_classification`` / ``make_regression`` so the
synthetic data follows a standard, well-understood generator. The grid
spans laptop-to-GPU regimes; override it per run with a config file or the
``-d`` CLI flag.
"""

import numpy as np
from benchopt import BaseDataset
from sklearn.datasets import make_classification, make_regression


class Dataset(BaseDataset):
    name = "Simulated"

    # Cross product over (n_samples, n_features), the task type, and the number
    # of classes (used only for classification). Each combination is a distinct
    # dataset instance. Override any subset from the CLI, e.g.
    #   -d "Simulated[n_samples=[1000,5000],task=classification]"
    parameters = {
        "n_samples, n_features": [
            (1000, 20),
            (5000, 50),
            (10000, 100),
            (50000, 100),
        ],
        "n_classes": [10, 100],
        "task": ["classification", "regression"],
    }

    test_parameters = {
        "n_samples": 200,
        "n_features": 5,
        "n_classes": 2,
        "task": "classification",
    }

    def get_data(self):
        # `use_repetition=True` so each repetition draws a different dataset,
        # while all solvers within a repetition see the same one.
        seed = self.get_seed(use_repetition=True)

        if self.task == "classification":
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=self.n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                class_sep=1.0,
                random_state=seed,
            )
            y = y.astype(np.int64)
        else:
            X, y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=self.n_features,
                noise=0.1,
                random_state=seed,
            )
            y = y.astype(np.float64)

        return dict(X=X, y=y, task=self.task, n_classes=self.n_classes)

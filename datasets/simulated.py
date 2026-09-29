"""Simulated tabular datasets for the TabICL benchmark.

Uses scikit-learn's ``make_classification`` / ``make_regression`` so the
synthetic data follows a standard, well-understood generator. The grid spans
laptop-to-GPU regimes; override it per run with a config file or the
``-d`` CLI flag.

``n_train_samples`` and ``n_features`` are independent parameters so the two
dataset-size axes can be swept separately — vary rows at fixed columns (and
vice-versa) to isolate the effect of each on fit/predict time and memory.

The dataset generates ``n_train_samples + n_test_samples`` total samples and
splits them into a train and a test set (the split uses a per-repetition
seed so different repetitions draw different splits, while all solvers within
a repetition see the same one). The split is done here, not in the objective.
"""

import numpy as np
from benchopt import BaseDataset
from sklearn.datasets import make_classification, make_regression


class Dataset(BaseDataset):
    name = "Simulated"

    # ``n_train_samples`` and ``n_features`` are INDEPENDENT parameters so the
    # two dataset-size axes can be swept separately (essential for scaling
    # plots: vary rows at fixed columns, and vice-versa). Override from the CLI
    # with ``-d`` and list a dataset more than once to run a union of grids,
    # e.g. a rows sweep + a columns sweep in one command:
    #   -d "Simulated[n_train_samples=[300,1000,2000,4000],n_features=50,...]"
    #   -d "Simulated[n_train_samples=1000,n_features=[20,40,100,200],...]"
    #
    # WARNING: the full default cartesian product below is large
    # (8 × 6 × 4 × 2 × 2 = 768 dataset instances). Always restrict with ``-d``
    # for a real run — the default grid is intentionally broad so any size the
    # benchmark has ever used stays selectable without a code change.
    parameters = {
        "n_train_samples": [100, 300, 1000, 2000, 4000, 5000, 10000, 50000],
        "n_features": [10, 20, 40, 50, 100, 200],
        "n_test_samples": [1, 32, 512, 2048],
        "n_classes": [10, 100],
        "task": ["classification", "regression"],
    }

    test_parameters = {
        "n_train_samples": 200,
        "n_features": 5,
        "n_test_samples": 50,
        "n_classes": 2,
        "task": "classification",
    }

    def get_data(self):
        # `use_repetition=True` so each repetition draws a different dataset,
        # while all solvers within a repetition see the same one.
        seed = self.get_seed(use_repetition=True)
        rng = np.random.default_rng(seed)

        n_total = self.n_train_samples + self.n_test_samples

        if self.task == "classification":
            X, y = make_classification(
                n_samples=n_total,
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
                n_samples=n_total,
                n_features=self.n_features,
                n_informative=self.n_features,
                noise=0.1,
                random_state=seed,
            )
            y = y.astype(np.float64)

        # Shuffle and split into train / test.
        perm = rng.permutation(n_total)
        train_idx = perm[: self.n_train_samples]
        test_idx = perm[self.n_train_samples :]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task=self.task,
            n_classes=self.n_classes,
        )

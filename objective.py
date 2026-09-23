"""Objective for the TabICL benchmark.

Meures, for a TabICL-based classifier or regressor on a tabular dataset:

* execution walltime, split into ``fit_time`` (data preparation) and
  ``predict_time`` (in-context learning forward pass),
* peak RAM (MB) and peak VRAM (MB) during ``fit`` + ``predict``,
* prediction quality (accuracy + log-loss for classification,
  RMSE + R2 for regression).

Benchopt already records the solver ``time`` column (walltime of
``Solver.run``) and rich hardware provenance (CPU model, core count, system
RAM, CUDA version, package versions) in every result parquet. The
``gpu_name``/``device`` columns captured here complement that with the
accelerator actually selected by TabICL, which benchopt does not record.
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from benchmark_utils.versions import PACKAGE_VERSIONS


class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "TabICL inference"

    # URL of the benchmark repository.
    url = "https://github.com/probabl/tabicl-benchmarks"

    min_benchopt_version = "1.10.0"

    # Fixed-budget ML benchmark: one call to Solver.run() per config, no
    # convergence curve. All solvers inherit this.
    sampling_strategy = "run_once"

    # A short free-text label describing *why* this run is performed. Must be
    # set from the CLI/config (it raises if left None) so every result parquet
    # carries a human-readable purpose. Recorded automatically by benchopt as
    # the ``p_objective_objective_label`` column.
    #
    # ``scratch_dir`` is a shared scratch location passed to all solvers;
    # solvers that need on-disk storage (e.g. disk offloading) create a
    # sub-directory inside it. Recorded as ``p_objective_scratch_dir``.
    parameters = {
        "objective_label": [None],
        "scratch_dir": [None],
    }

    # Tiny config exercising the objective->dataset->solver chain fast.
    test_config = {
        "dataset": {
            "name": "simulated",
            "n_samples": 200,
            "n_features": 5,
            "n_classes": 2,
            "task": "classification",
        },
        "objective_label": "test",
        "scratch_dir": None,
    }

    def set_data(self, X, y, task, n_classes):
        """Store the dataset and build a deterministic train/test split.

        The split uses ``self.get_seed(use_repetition=True)`` so each
        repetition draws a different split, while all solvers within a
        repetition see the *same* split (fair comparison).

        ``n_classes`` is the dataset *parameter* (10 or 100), kept as-is so
        solvers can use it to skip redundant regression instances. The actual
        class count for scoring is derived from the data for classification.
        """
        if self.objective_label is None:
            raise ValueError(
                "objective_label must be set via the CLI or config file, e.g. "
                "-o \"TabICL inference[objective_label='baseline CPU run']\". "
                "Provide a short sentence describing why this run is performed; "
                "it is recorded as `p_objective_objective_label` in the output."
            )
        self.task = task
        self.n_classes = n_classes  # dataset parameter, passed to solvers
        if task == "classification":
            self.n_classes_actual = int(np.unique(y).max()) + 1
        else:
            self.n_classes_actual = None

        rng = np.random.default_rng(self.get_seed(use_repetition=True))
        n = X.shape[0]
        perm = rng.permutation(n)
        n_test = max(1, n // 5)  # 20% held-out test set
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        self.X_train = X[train_idx]
        self.X_test = X[test_idx]
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]

    def skip(self, X, y, task, n_classes):
        # No objective-level skip; the redundant regression+n_classes combos
        # are filtered at the solver level (see solvers' ``skip``), which is
        # the flow benchopt's test suite expects.
        return False, None

    def evaluate_result(
        self,
        y_pred,
        y_score=None,
        y_encoder=None,
        fit_time=0.0,
        predict_time=0.0,
        ram_peak_mb=0.0,
        vram_peak_mb=0.0,
        disk_offload_dir=None,
        device="cpu",
        gpu_name=None,
    ):
        """Score a solver's predictions and attach resource metrics.

        For classification the solver returns probabilities (``y_score``) and
        the fitted label encoder (``y_encoder``); the argmax + inverse
        transform to recover ``y_pred`` is done here, untimed, so it does not
        pollute the solver's prediction timing.

        ``device`` and ``gpu_name`` are resolved by the solver for the device
        it actually used (not re-detected here), so a CPU run on a GPU machine
        correctly records ``device='cpu'``, ``gpu_name=None``.

        ``value`` is the quantity benchopt monitors/plots by default; it is
        defined as "lower is better" (``1 - accuracy`` for classification,
        ``rmse`` for regression).
        """
        provenance = {
            "fit_time": fit_time,
            "predict_time": predict_time,
            "ram_peak_mb": ram_peak_mb,
            "vram_peak_mb": vram_peak_mb,
            "disk_offload_dir": disk_offload_dir,
            "gpu_name": gpu_name,
            "device": device,
            **PACKAGE_VERSIONS,
        }
        if self.task == "classification":
            # Derive class labels from probabilities (untimed scoring step).
            if y_pred is None and y_score is not None and y_encoder is not None:
                y_pred = y_encoder.inverse_transform(np.argmax(y_score, axis=1))
            acc = accuracy_score(self.y_test, y_pred)
            if y_score is not None:
                ll = log_loss(
                    self.y_test, y_score, labels=list(range(self.n_classes_actual))
                )
            else:
                ll = float("nan")
            return dict(
                value=1.0 - acc,
                accuracy=acc,
                log_loss=ll,
                **provenance,
            )
        else:
            r = float(np.sqrt(mean_squared_error(self.y_test, y_pred)))
            return dict(
                value=r,
                rmse=r,
                r2=r2_score(self.y_test, y_pred),
                **provenance,
            )

    def get_one_result(self):
        """Return a dummy result for `benchopt test` (no real solver needed)."""
        n_test = self.y_test.shape[0]
        if self.task == "classification":
            y_pred = np.zeros(n_test, dtype=self.y_test.dtype)
            y_score = np.ones((n_test, self.n_classes_actual)) / self.n_classes_actual
            return dict(
                y_pred=y_pred,
                y_score=y_score,
                y_encoder=None,
                disk_offload_dir=None,
                device="cpu",
                gpu_name=None,
            )
        return dict(
            y_pred=np.zeros(n_test), disk_offload_dir=None, device="cpu", gpu_name=None
        )

    def get_objective(self):
        """Payload handed to every solver via ``set_objective``.

        ``y_test`` is kept on the objective for scoring and never exposed
        to solvers. ``scratch_dir`` is forwarded so solvers that need disk
        space (e.g. disk offloading) can use it.
        """
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task=self.task,
            n_classes=self.n_classes,
            scratch_dir=self.scratch_dir,
        )

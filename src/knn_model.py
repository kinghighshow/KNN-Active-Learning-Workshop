"""KNN training and evaluation steps of the KNN Workshop pipeline.

Pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator

This module holds the last two steps:
    KNNModel   splits the data, tunes k, trains the classifier and predicts.
    Evaluator  computes metrics, compares against a baseline and builds the charts.

Charts use Plotly (team convention). Plotly is imported inside the chart
methods only, so the modelling code also runs where Plotly is not installed.

Owner: Antonio Sainz (KNN Implementation & Evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42


@dataclass
class KNNModel:
    """Split, tune, train and predict with scikit-learn's KNeighborsClassifier.

    k is chosen with stratified cross-validation on the TRAINING set only,
    scored by macro F1, so the test set stays untouched until the final check.
    """

    k_values: range = field(default_factory=lambda: range(3, 26))
    metrics: tuple = ("euclidean", "manhattan")
    cv_folds: int = 5
    scoring: str = "f1_macro"
    random_state: int = RANDOM_STATE

    best_k: int | None = None
    best_metric: str | None = None
    tuning_results: pd.DataFrame | None = None
    model: KNeighborsClassifier | None = None

    def split(self, X, y, test_size: float = 0.2):
        """80/20 split, stratified so rare classes (Rain, Snow) appear in both sets."""
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

    def tune(self, X_train, y_train) -> pd.DataFrame:
        """Cross-validate every (k, distance metric) pair on the training set."""
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        rows = []
        for metric in self.metrics:
            for k in self.k_values:
                scores = cross_val_score(
                    KNeighborsClassifier(n_neighbors=k, metric=metric),
                    X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=-1,
                )
                rows.append({"k": k, "metric": metric,
                             "cv_mean": scores.mean(), "cv_std": scores.std()})
        self.tuning_results = pd.DataFrame(rows)
        best = self.tuning_results.loc[self.tuning_results["cv_mean"].idxmax()]
        self.best_k, self.best_metric = int(best["k"]), str(best["metric"])
        return self.tuning_results

    def fit(self, X_train, y_train, k: int | None = None, metric: str | None = None):
        """Train the final model (defaults to the tuned k and metric, else k=5)."""
        k = k or self.best_k or 5
        metric = metric or self.best_metric or "euclidean"
        self.model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)


class Evaluator:
    """Metrics, baseline comparison and Plotly charts for a fitted classifier."""

    def __init__(self, labels=None):
        self.labels = labels

    @staticmethod
    def metrics(y_true, y_pred) -> dict:
        """Accuracy plus macro precision / recall / F1 (classes are imbalanced)."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def report(self, y_true, y_pred) -> pd.DataFrame:
        """Per-class precision, recall, F1 and support as a table."""
        rep = classification_report(y_true, y_pred, labels=self.labels,
                                    output_dict=True, zero_division=0)
        return pd.DataFrame(rep).T.round(3)

    def confusion(self, y_true, y_pred) -> pd.DataFrame:
        labels = self.labels if self.labels is not None else sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(cm, index=[f"true {l}" for l in labels],
                            columns=[f"pred {l}" for l in labels])

    def baseline(self, X_train, y_train, X_test, y_test) -> dict:
        """Majority-class baseline: what KNN has to beat to be useful."""
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        return self.metrics(y_test, dummy.predict(X_test))

    def compare(self, results: dict) -> pd.DataFrame:
        """Side-by-side table of metric dicts, e.g. {'Baseline': {...}, 'KNN': {...}}."""
        return pd.DataFrame(results).T.round(3)

    # ---------- Plotly charts ----------

    @staticmethod
    def plot_k_tuning(tuning_results: pd.DataFrame, best_k: int, best_metric: str,
                      title: str = "Choosing k: cross-validated macro F1"):
        import plotly.graph_objects as go

        fig = go.Figure()
        for metric, part in tuning_results.groupby("metric"):
            fig.add_trace(go.Scatter(
                x=part["k"], y=part["cv_mean"], mode="lines+markers", name=metric,
                error_y=dict(type="data", array=part["cv_std"], visible=True, thickness=1),
            ))
        best = tuning_results.query("k == @best_k and metric == @best_metric").iloc[0]
        fig.add_trace(go.Scatter(
            x=[best_k], y=[best["cv_mean"]], mode="markers", name=f"best (k={best_k})",
            marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        ))
        fig.update_layout(title=title, xaxis_title="k (number of neighbours)",
                          yaxis_title="Mean macro F1 (5-fold CV)",
                          template="plotly_white", legend_title="Distance")
        return fig

    def plot_confusion(self, y_true, y_pred, title: str = "Confusion matrix (test set)"):
        import plotly.express as px

        labels = self.labels if self.labels is not None else sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicted", y="Actual", color="Hours"))
        fig.update_layout(title=title, template="plotly_white")
        return fig

    @staticmethod
    def plot_metric_comparison(comparison: pd.DataFrame,
                               title: str = "KNN vs majority-class baseline"):
        import plotly.express as px

        long = comparison.reset_index(names="model").melt(
            id_vars="model", var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                     text_auto=".2f", range_y=[0, 1])
        fig.update_layout(title=title, template="plotly_white")
        return fig

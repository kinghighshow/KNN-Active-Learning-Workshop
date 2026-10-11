"""Steps 3-4, KNN training (k and distance tuning) and evaluation (metrics, baseline, charts). Owner: Antonio.

Part of the KNN Workshop pipeline: DataLoader -> Preprocessor -> KNNModel -> Evaluator,
chained by KNNPipeline. Used by KNN_Workshop_Solution.ipynb.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src.config import RANDOM_STATE


class KNNModel:
    def __init__(self, k_values=range(3, 26), metrics=("euclidean", "manhattan"), cv_folds=5):
        self.k_values = k_values
        self.metrics = metrics
        self.cv_folds = cv_folds
        self.best_k = None
        self.best_metric = None
        self.model = None

    def tune(self, X_train, y_train):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        rows = []
        for metric in self.metrics:
            for k in self.k_values:
                scores = cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric),
                                         X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
                rows.append({"k": k, "metric": metric, "cv_mean": scores.mean(), "cv_std": scores.std()})
        self.tuning_results = pd.DataFrame(rows)
        best = self.tuning_results.loc[self.tuning_results["cv_mean"].idxmax()]
        self.best_k, self.best_metric = int(best["k"]), best["metric"]
        return self.tuning_results

    def fit(self, X_train, y_train):
        self.model = KNeighborsClassifier(n_neighbors=self.best_k or 5,
                                          metric=self.best_metric or "euclidean")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


class Evaluator:
    def __init__(self, labels=None):
        self.labels = labels

    @staticmethod
    def metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def baseline(self, y_train, y_test):
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
        return self.metrics(y_test, dummy.predict(np.zeros((len(y_test), 1))))

    @staticmethod
    def compare(results):
        return pd.DataFrame(results).T.round(3)

    def report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, labels=self.labels,
                                       output_dict=True, zero_division=0)
        return pd.DataFrame(report).T.round(3)

    def plot_k_tuning(self, tuning, best_k, best_metric):
        fig = go.Figure()
        for metric, part in tuning.groupby("metric"):
            fig.add_trace(go.Scatter(x=part["k"], y=part["cv_mean"], mode="lines+markers", name=metric,
                                     error_y=dict(type="data", array=part["cv_std"], thickness=1)))
        best = tuning[(tuning["k"] == best_k) & (tuning["metric"] == best_metric)].iloc[0]
        fig.add_trace(go.Scatter(x=[best_k], y=[best["cv_mean"]], mode="markers", name=f"best (k={best_k})",
                                 marker=dict(size=16, symbol="star", line=dict(width=1, color="black"))))
        fig.update_layout(title="Choosing k: cross-validated macro F1 (training set)",
                          xaxis_title="k (number of neighbours)", yaxis_title="Mean macro F1 (5-fold CV)",
                          legend_title="Distance", template="plotly_white")
        return fig

    def plot_comparison(self, comparison):
        long = comparison.reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="score")
        return px.bar(long, x="metric", y="score", color="model", barmode="group", text_auto=".2f",
                      range_y=[0, 1], title="KNN vs majority-class baseline (test set)",
                      template="plotly_white")

    def plot_confusion(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        return px.imshow(cm, x=self.labels, y=self.labels, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual", color="Hours"),
                         title="Confusion matrix (test set)", template="plotly_white")

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _ensure_parent_dir(savepath):
    if savepath is None:
        return
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame, original_col="n", log_col="n_log", savepath=None):
    """
    Plot the original and log-transformed target distributions.
    """
    if original_col not in df.columns or log_col not in df.columns:
        raise KeyError(f"Both '{original_col}' and '{log_col}' must exist in dataframe.")

    _ensure_parent_dir(savepath)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].hist(df[original_col], bins=30)
    ax[0].set_title("Original target distribution")
    ax[0].set_xlabel(original_col)
    ax[0].set_ylabel("Count")

    ax[1].hist(df[log_col], bins=30)
    ax[1].set_title("Log-transformed target distribution")
    ax[1].set_xlabel(log_col)
    ax[1].set_ylabel("Count")

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def make_prediction_plot(y_true, y_pred, title="Prediction plot", savepath=None):
    """
    Plot predicted values against true values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _ensure_parent_dir(savepath)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r-")

    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def evaluate_on_test(models, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate fitted models on the held-out test set.
    """
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        rows.append({
            "Model": name,
            "Test_MAE": float(mean_absolute_error(y_test, y_pred)),
            "Test_RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "Test_R2": float(r2_score(y_test, y_pred)),
        })

    return pd.DataFrame(rows).sort_values("Test_RMSE").reset_index(drop=True)


def save_model_comparison(cv_results: pd.DataFrame, test_results: pd.DataFrame, savepath="results/model_comparison.csv"):
    """
    Merge CV and test-set results and save to CSV.
    """
    _ensure_parent_dir(savepath)

    comparison = cv_results.merge(test_results, on="Model", how="inner")
    comparison.to_csv(savepath, index=False)
    return comparison

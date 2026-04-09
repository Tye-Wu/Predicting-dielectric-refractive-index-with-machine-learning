import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_models(random_state: int = 42):
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state
        ),
    }


def cross_validate_models(models, X_train, y_train, cv: int = 5):
    rows = []

    for name, model in models.items():
        mae_scores = -cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="neg_mean_absolute_error"
        )
        mse_scores = -cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="neg_mean_squared_error"
        )
        r2_scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="r2"
        )

        rows.append({
            "Model": name,
            "CV_MAE": np.mean(mae_scores),
            "CV_RMSE": np.mean(np.sqrt(mse_scores)),
            "CV_R2": np.mean(r2_scores),
        })

    return pd.DataFrame(rows).sort_values("CV_RMSE")


def tune_models(X_train, y_train, random_state: int = 42):
    tuned_models = {}

    xgb_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 4],
    }

    hgb_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [None, 5, 10],
        "max_iter": [100, 200],
    }

    xgb_search = GridSearchCV(
        XGBRegressor(random_state=random_state),
        xgb_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    tuned_models["XGBoost"] = xgb_search.best_estimator_

    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        rf_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    tuned_models["RandomForest"] = rf_search.best_estimator_

    hgb_search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        hgb_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    hgb_search.fit(X_train, y_train)
    tuned_models["HistGradientBoosting"] = hgb_search.best_estimator_

    tuned_models["LinearRegression"] = LinearRegression()

    return tuned_models

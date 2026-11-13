from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT       = Path(__file__).resolve().parent.parent
MLFLOW_DB  = ROOT / "mlflow.db"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scorers built-in: manejan predict_proba internamente, no necesitan make_scorer
SCORERS = {
    "auc":    "roc_auc",
    "pr_auc": "average_precision",
}


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, cv=CV) -> dict:
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORERS, n_jobs=-1)
    return {
        "auc_mean":    scores["test_auc"].mean(),
        "auc_std":     scores["test_auc"].std(),
        "pr_auc_mean": scores["test_pr_auc"].mean(),
        "pr_auc_std":  scores["test_pr_auc"].std(),
        "gini_mean":   2 * scores["test_auc"].mean() - 1,
    }


def log_run(run_name: str, model, X: pd.DataFrame, y: pd.Series,
            feature_groups: list[str], extra_params: dict | None = None) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("credit_scoring_features")

    metrics = evaluate_cv(model, X, y)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model":      type(model).__name__,
            "groups":     ",".join(feature_groups) or "baseline",
            "n_features": X.shape[1],
            "cv_folds":   CV.get_n_splits(),
            **(extra_params or {}),
        })
        mlflow.log_metrics(metrics)
    return metrics


def baseline_model() -> Pipeline:
    """Pipeline: imputer (mediana, fit sobre train) + Random Forest.
    El imputer dentro del pipeline evita leakage y permite NaN en X."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_depth=None,
            n_jobs=-1, random_state=42,
        )),
    ])


def baseline_logit() -> Pipeline:
    """Pipeline: imputer (mediana) + scaler + Logistic Regression.
    Scaler necesario: la logística asume features en escala comparable.
    class_weight=None: baseline natural, sin resampling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=42,
        )),
    ])
"""
SHAP-based explainability for the Financial Health Card.

XGBoost + shap.TreeExplainer gives exact, fast (no sampling/kernel approx)
Shapley values for every model in this project, so the same helper works for
the overall score regressor, the sub-score regressors, and the risk/eligibility
classifiers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=None)
def _explainer_for(model_filename: str):
    model = joblib.load(MODELS_DIR / model_filename)
    return shap.TreeExplainer(model), model


def explain_regressor(model_filename: str, X_row: pd.DataFrame, top_n: int = 6) -> dict:
    """Returns the prediction plus the top_n features pushing it up/down."""
    explainer, model = _explainer_for(model_filename)
    shap_values = explainer.shap_values(X_row)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])
    else:
        base_value = float(base_value)

    contributions = pd.Series(shap_values[0], index=X_row.columns).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    top = contributions.head(top_n)
    drivers = [
        {
            "feature": feat,
            "value": _readable_value(X_row.iloc[0][feat]),
            "impact": round(float(val), 3),
            "direction": "increases" if val > 0 else "decreases",
        }
        for feat, val in top.items()
    ]
    prediction = float(model.predict(X_row)[0])
    return {"prediction": round(prediction, 2), "base_value": round(base_value, 2), "top_drivers": drivers}


def explain_classifier(model_filename: str, X_row: pd.DataFrame, class_names: list[str], top_n: int = 6) -> dict:
    explainer, model = _explainer_for(model_filename)
    proba = model.predict_proba(X_row)[0]
    pred_class_idx = int(np.argmax(proba))
    shap_values = explainer.shap_values(X_row)

    # shap for multiclass XGBoost via TreeExplainer returns array shaped
    # (n_samples, n_features, n_classes) in recent versions.
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        class_shap = sv[0, :, pred_class_idx]
    else:
        class_shap = sv[pred_class_idx][0] if isinstance(shap_values, list) else sv[0]

    contributions = pd.Series(class_shap, index=X_row.columns).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    top = contributions.head(top_n)
    drivers = [
        {
            "feature": feat,
            "value": _readable_value(X_row.iloc[0][feat]),
            "impact": round(float(val), 3),
            "direction": "increases" if val > 0 else "decreases",
        }
        for feat, val in top.items()
    ]
    return {
        "predicted_class": class_names[pred_class_idx],
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(class_names, proba)},
        "top_drivers": drivers,
    }


def _readable_value(v) -> float | int | str:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 3)
    return v

"""
Trains every model behind the Financial Health Card:

  - 6 sub-score regressors        (Business_Stability, Cashflow, Revenue_Consistency,
                                    Payment_Behaviour, Business_Growth, Compliance)
  - 1 overall score regressor     (Financial_Health_Score, 0-100)
  - 1 probability-of-default regressor
  - 1 credit-risk classifier      (Low / Medium / High)
  - 1 credit-eligibility classifier (Yes / No)
  - 1 credit-limit regressor      (trained on the eligible subset, log1p target)

All models are gradient-boosted trees (XGBoost) so a single SHAP TreeExplainer
strategy (see src/explain.py) works uniformly across the whole card.

Run: python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    r2_score, root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.data_pipeline import (
    CREDIT_LIMIT_TARGET, ELIGIBLE_TARGET, OVERALL_SCORE_TARGET, PD_TARGET,
    RISK_TARGET, SUBSCORE_TARGETS, build_feature_matrix, load_raw, save_feature_schema,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
RANDOM_STATE = 42

RISK_CLASSES = ["Low", "Medium", "High"]  # fixed, ordered by increasing risk


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
    }


def _clf_metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = load_raw()
    X, dummy_columns = build_feature_matrix(df)
    save_feature_schema(dummy_columns)

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df[RISK_TARGET]
    )
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]

    report = {"n_train": len(idx_train), "n_test": len(idx_test), "n_features": X.shape[1]}

    # ---- 1. sub-score regressors ----------------------------------------
    report["subscores"] = {}
    for target in SUBSCORE_TARGETS:
        print(f"Training sub-score regressor: {target}")
        model = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            n_jobs=-1, reg_lambda=1.0,
        )
        model.fit(X_train, df_train[target])
        preds = model.predict(X_test)
        report["subscores"][target] = _reg_metrics(df_test[target], preds)
        joblib.dump(model, MODELS_DIR / f"subscore_{target}.joblib")

    # ---- 2. overall financial health score -------------------------------
    print("Training overall Financial_Health_Score regressor")
    overall_model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
    )
    overall_model.fit(X_train, df_train[OVERALL_SCORE_TARGET])
    overall_preds = overall_model.predict(X_test)
    report["overall_score"] = _reg_metrics(df_test[OVERALL_SCORE_TARGET], overall_preds)
    joblib.dump(overall_model, MODELS_DIR / "overall_score.joblib")

    # ---- 3. probability of default ---------------------------------------
    print("Training Probability_of_Default regressor")
    pd_model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
    )
    pd_model.fit(X_train, df_train[PD_TARGET])
    pd_preds = np.clip(pd_model.predict(X_test), 0, 1)
    report["probability_of_default"] = _reg_metrics(df_test[PD_TARGET], pd_preds)
    joblib.dump(pd_model, MODELS_DIR / "pd_model.joblib")

    # ---- 4. credit risk category classifier -------------------------------
    print("Training Credit_Risk_Category classifier")
    risk_y_train = df_train[RISK_TARGET].map({c: i for i, c in enumerate(RISK_CLASSES)})
    risk_y_test = df_test[RISK_TARGET].map({c: i for i, c in enumerate(RISK_CLASSES)})
    risk_model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
        n_jobs=-1, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    )
    risk_model.fit(X_train, risk_y_train)
    risk_preds = risk_model.predict(X_test)
    report["risk_category"] = _clf_metrics(risk_y_test, risk_preds, labels=[0, 1, 2])
    report["risk_category"]["labels"] = RISK_CLASSES
    joblib.dump(risk_model, MODELS_DIR / "risk_classifier.joblib")
    with open(MODELS_DIR / "risk_classes.json", "w") as f:
        json.dump(RISK_CLASSES, f)

    # ---- 5. credit eligibility classifier ----------------------------------
    print("Training Credit_Eligible classifier")
    elig_y_train = (df_train[ELIGIBLE_TARGET] == "Yes").astype(int)
    elig_y_test = (df_test[ELIGIBLE_TARGET] == "Yes").astype(int)
    elig_model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
        n_jobs=-1, eval_metric="logloss",
    )
    elig_model.fit(X_train, elig_y_train)
    elig_preds = elig_model.predict(X_test)
    report["eligibility"] = _clf_metrics(elig_y_test, elig_preds, labels=[0, 1])
    report["eligibility"]["labels"] = ["No", "Yes"]
    joblib.dump(elig_model, MODELS_DIR / "eligible_classifier.joblib")

    # ---- 6. recommended credit limit (trained on eligible subset) ---------
    print("Training Recommended_Credit_Limit_INR regressor (eligible subset, log1p target)")
    train_elig_mask = df_train[ELIGIBLE_TARGET] == "Yes"
    test_elig_mask = df_test[ELIGIBLE_TARGET] == "Yes"
    limit_model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
    )
    limit_model.fit(X_train[train_elig_mask], np.log1p(df_train.loc[train_elig_mask, CREDIT_LIMIT_TARGET]))
    limit_preds = np.expm1(limit_model.predict(X_test[test_elig_mask]))
    report["credit_limit_eligible_only"] = _reg_metrics(
        df_test.loc[test_elig_mask, CREDIT_LIMIT_TARGET], limit_preds
    )
    joblib.dump(limit_model, MODELS_DIR / "credit_limit_model.joblib")

    # ---- feature importance snapshot for the overall score model ----------
    importances = pd.Series(overall_model.feature_importances_, index=dummy_columns)
    top_features = importances.sort_values(ascending=False).head(15).round(4).to_dict()
    report["overall_score_top_features"] = top_features

    with open(MODELS_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    _write_markdown_report(report)
    print("Done. Metrics written to models/metrics_report.json and docs/MODEL_REPORT.md")


def _write_markdown_report(report: dict) -> None:
    lines = ["# Model Report", "", "Auto-generated by `python -m src.train`. All models are XGBoost", "gradient-boosted trees trained on an 80/20 split (`random_state=42`),", f"stratified by `Credit_Risk_Category` — {report['n_train']} train / {report['n_test']} test rows, {report['n_features']} input features.", ""]

    lines += ["## Sub-score regressors (0-100 scale)", "", "| Sub-score | R² | MAE | RMSE |", "|---|---|---|---|"]
    for target, m in report["subscores"].items():
        lines.append(f"| {target} | {m['r2']} | {m['mae']} | {m['rmse']} |")
    lines.append("")

    m = report["overall_score"]
    lines += ["## Overall Financial Health Score", "", f"R² = {m['r2']}, MAE = {m['mae']}, RMSE = {m['rmse']}", ""]

    m = report["probability_of_default"]
    lines += ["## Probability of Default", "", f"R² = {m['r2']}, MAE = {m['mae']}, RMSE = {m['rmse']}", ""]

    m = report["risk_category"]
    lines += [
        "## Credit Risk Category classifier (Low / Medium / High)", "",
        f"Accuracy = {m['accuracy']}, Macro F1 = {m['f1_macro']}", "",
        "Confusion matrix (rows = actual, cols = predicted, order Low/Medium/High):", "```",
    ]
    for row in m["confusion_matrix"]:
        lines.append(str(row))
    lines += ["```", ""]

    m = report["eligibility"]
    lines += [
        "## Credit Eligibility classifier (Yes / No)", "",
        f"Accuracy = {m['accuracy']}, Macro F1 = {m['f1_macro']}", "",
        "Confusion matrix (rows = actual, cols = predicted, order No/Yes):", "```",
    ]
    for row in m["confusion_matrix"]:
        lines.append(str(row))
    lines += ["```", ""]

    m = report["credit_limit_eligible_only"]
    lines += ["## Recommended Credit Limit (eligible MSMEs only, INR)", "", f"R² = {m['r2']}, MAE = ₹{m['mae']:,.0f}, RMSE = ₹{m['rmse']:,.0f}", ""]

    lines += ["## Top 15 drivers of the Overall Financial Health Score", "", "(XGBoost gain-based feature importance — see `docs/MODEL_REPORT.md` SHAP", "notes in the explainability section for per-record attributions.)", "", "| Feature | Importance |", "|---|---|"]
    for feat, imp in report["overall_score_top_features"].items():
        lines.append(f"| {feat} | {imp} |")
    lines.append("")

    lines += [
        "## Explainability methodology", "",
        "Every model above is a gradient-boosted tree ensemble (XGBoost), so a single",
        "`shap.TreeExplainer` per model gives exact (not sampled/approximated) Shapley",
        "values — see `src/explain.py`. For any single MSME, the overall score, the",
        "risk-category decision, and the eligibility decision can each be decomposed into",
        "a base value (population average) plus signed per-feature contributions that sum",
        "exactly to the model's prediction. This is what powers the \"Explainability — why",
        "this score\" panel in the dashboard and the `/api/v1/explain/{msme_id}` endpoint.",
        "SHAP is computed on demand per record (not batched for all 50k) since it is the",
        "one part of the pipeline that doesn't need to scale to the full population at once.",
        "",
        "## Anomaly detection methodology", "",
        "`src/anomaly.py` trains an unsupervised `IsolationForest` (300 trees,",
        "3% contamination) over 13 transactional/behavioural features — transaction",
        "volatility, cashflow stability, overdraft usage, GST filing timeliness, EMI",
        "on-time rate, salary consistency, employee attrition, invoice delay, customer",
        "concentration, vendor timeliness, revenue growth, UPI ticket size and daily",
        "transaction count. There is no labelled anomaly/fraud column in the data (nor",
        "would there be in production), so this deliberately stays unsupervised: it flags", "the ~3% of MSMEs whose alternative-data pattern sits furthest from the rest of the",
        "population, which is what an underwriter actually wants surfaced for manual review",
        "— see `python -m src.anomaly` output for the flagged count on the full dataset.",
        "",
    ]

    with open(DOCS_DIR / "MODEL_REPORT.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

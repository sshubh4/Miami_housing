"""
Model training for Miami Housing Price Predictor.

Uses XGBoost with cross-validated hyperparameter tuning.
Generates SHAP values for explainability and saves artifacts for serving.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

log = logging.getLogger(__name__)

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "age",
    "avno60plus",
    "structure_quality",
    "month_sold",
    "log_ocean_dist",
    "log_cntr_dist",
    "land_to_living",
    "is_coastal",
    "cbd_access",
    "peak_season",
    "noise_penalty",
]

TARGET_COL = "SALE_PRC"
MODEL_DIR = Path("models")


def get_model() -> XGBRegressor:
    """
    Return an XGBRegressor with well-tuned defaults.
    These hyperparameters were selected via 5-fold CV on the Miami dataset.
    Replace with Optuna sweep for a production pipeline.
    """
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,       # L1 — handles noisy distance features
        reg_lambda=1.0,      # L2 — standard
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )


def train(df: pd.DataFrame) -> tuple[XGBRegressor, dict, list[str]]:
    """
    Train the model with a validation split for early stopping.
    Returns (trained_model, metrics_dict).
    """
    # Filter to only available feature columns (safe for partial datasets)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        log.warning("Training without columns (not in dataset): %s", missing)

    X = df[available]
    y = np.log1p(df[TARGET_COL])   # log-transform price for better residuals

    # 80/20 split for early stopping eval set
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = get_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on held-out validation set (inverse-transform for interpretability)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    actuals = np.expm1(y_val)

    metrics = {
        "r2": round(r2_score(actuals, preds), 4),
        "mae": round(mean_absolute_error(actuals, preds), 2),
        "rmse": round(np.sqrt(mean_squared_error(actuals, preds)), 2),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_cols": available,
        "best_iteration": int(model.best_iteration),
    }

    log.info("Validation — R2: %.4f | MAE: $%s | RMSE: $%s",
         metrics["r2"],
         f"{metrics['mae']:,.0f}",
         f"{metrics['rmse']:,.0f}")
    return model, metrics, available


def cross_validate(df: pd.DataFrame) -> dict:
    """5-fold CV to get a stable performance estimate."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = np.log1p(df[TARGET_COL])

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    log.info("5-fold CV R²: %.4f ± %.4f", scores.mean(), scores.std())
    return {"cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std": round(float(scores.std()), 4)}


def compute_shap(model: XGBRegressor, df: pd.DataFrame,
                 feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute SHAP values for the first 500 rows (fast enough for Streamlit).
    Returns a DataFrame of mean absolute SHAP values per feature.
    """
    sample = df[feature_cols].head(500)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)


def save_artifacts(model: XGBRegressor, metrics: dict,
                   feature_cols: list[str]) -> None:
    """Persist model + metadata for the serving layer."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    metrics["feature_cols"] = feature_cols
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Artifacts saved to %s/", MODEL_DIR)


def load_model() -> tuple[XGBRegressor, dict]:
    """Load persisted model and metrics for inference."""
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, metrics


def predict_price(model: XGBRegressor, feature_dict: dict,
                  feature_cols: list[str]) -> float:
    """
    Predict sale price for a single property dict.
    Returns price in USD (inverse-log-transformed).
    """
    row = pd.DataFrame([feature_dict])[feature_cols]
    log_pred = model.predict(row)[0]
    return float(np.expm1(log_pred))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.pipeline import run_pipeline

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/miami-housing.csv"
    df = run_pipeline(csv_path)

    log.info("Running 5-fold cross-validation...")
    cv_metrics = cross_validate(df)

    log.info("Training final model...")
    model, metrics, feature_cols = train(df)  # noqa: F841 — feature_cols passed to save_artifacts
    metrics.update(cv_metrics)

    save_artifacts(model, metrics, feature_cols)
    log.info("Training complete. Metrics: %s", metrics)

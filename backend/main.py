"""
Miami Housing Price Predictor — FastAPI backend.

Endpoints:
  GET  /api/data                — sampled property points + model metadata + SHAP
  POST /api/predict             — predict price for a lat/lon + property features
  GET  /api/neighbourhood-stats — median price, count, bounds per neighbourhood
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.pipeline import run_pipeline  # noqa: E402
from src.model import compute_shap, predict_price  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading pipeline and model…")
    df = run_pipeline(str(ROOT / "data" / "miami-housing.csv"))

    model = joblib.load(ROOT / "models" / "xgb_model.pkl")
    with open(ROOT / "models" / "metrics.json") as f:
        metrics = json.load(f)

    feature_cols: list[str] = metrics["feature_cols"]
    shap_df = compute_shap(model, df, feature_cols)

    _state.update(
        df=df,
        model=model,
        metrics=metrics,
        feature_cols=feature_cols,
        shap=shap_df,
    )
    log.info("Ready — %d properties loaded.", len(df))
    yield


app = FastAPI(title="Miami Housing API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
OCEAN_LAT, OCEAN_LON = 25.77, -80.13
CBD_LAT, CBD_LON = 25.775, -80.194


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# ── GET /api/data ─────────────────────────────────────────────────────────────
@app.get("/api/data")
def get_data():
    df: pd.DataFrame = _state["df"]
    metrics: dict = _state["metrics"]
    shap_df: pd.DataFrame = _state["shap"]

    sample = df.sample(min(1500, len(df)), random_state=42)
    cols = ["LATITUDE", "LONGITUDE", "SALE_PRC", "neighbourhood",
            "TOT_LVG_AREA", "age", "price_per_sqft"]
    points = sample[cols].fillna(0).round(4).to_dict(orient="records")

    shap_records = shap_df.rename(
        columns={"feature": "feature", "mean_abs_shap": "value"}
    ).to_dict(orient="records")

    return {
        "points": points,
        "total": int(len(df)),
        "median_price": float(df["SALE_PRC"].median()),
        "metrics": {
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "cv_r2_mean": metrics.get("cv_r2_mean"),
        },
        "shap": shap_records,
    }


# ── POST /api/predict ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    lat: float = Field(..., ge=25.1, le=26.0)
    lon: float = Field(..., ge=-80.9, le=-80.0)
    living_area: int = Field(1800, ge=100, le=20000)
    land_area: int = Field(7500, ge=500, le=100000)
    age: int = Field(30, ge=0, le=120)
    structure_quality: int = Field(3, ge=1, le=5)
    month_sold: int = Field(3, ge=1, le=12)
    aircraft_noise: bool = False


@app.post("/api/predict")
def predict(req: PredictRequest):
    df: pd.DataFrame = _state["df"]
    model = _state["model"]
    metrics: dict = _state["metrics"]
    feature_cols: list[str] = _state["feature_cols"]

    ocean_dist = haversine(req.lat, req.lon, OCEAN_LAT, OCEAN_LON)
    cntr_dist = haversine(req.lat, req.lon, CBD_LAT, CBD_LON)

    feature_dict = {
        "LND_SQFOOT": float(req.land_area),
        "TOT_LVG_AREA": float(req.living_area),
        "SPEC_FEAT_VAL": float(df["SPEC_FEAT_VAL"].median()),
        "RAIL_DIST": float(df["RAIL_DIST"].median()),
        "OCEAN_DIST": ocean_dist,
        "WATER_DIST": float(df["WATER_DIST"].median()),
        "CNTR_DIST": cntr_dist,
        "SUBCNTR_DI": float(df["SUBCNTR_DI"].median()),
        "HWY_DIST": float(df["HWY_DIST"].median()),
        "age": float(req.age),
        "avno60plus": float(int(req.aircraft_noise)),
        "structure_quality": float(req.structure_quality),
        "month_sold": float(req.month_sold),
        "log_ocean_dist": float(np.log1p(ocean_dist)),
        "log_cntr_dist": float(np.log1p(cntr_dist)),
        "land_to_living": float(req.land_area) / max(float(req.living_area), 1),
        "is_coastal": float(int(ocean_dist < 1000)),
        "cbd_access": 1.0 / (1.0 + cntr_dist / 1000),
        "peak_season": float(int(req.month_sold in [1, 2, 3, 4, 5])),
        "noise_penalty": float(int(req.aircraft_noise)) * ocean_dist,
    }

    predicted = predict_price(model, feature_dict, feature_cols)
    rmse = metrics["rmse"]

    nearby = df.copy()
    nearby["_dist"] = nearby.apply(
        lambda r: haversine(req.lat, req.lon, r["LATITUDE"], r["LONGITUDE"]), axis=1
    )
    comps = (
        nearby.nsmallest(5, "_dist")[
            ["_dist", "SALE_PRC", "TOT_LVG_AREA", "age", "neighbourhood"]
        ]
        .rename(columns={"_dist": "distance_m", "SALE_PRC": "sale_price",
                          "TOT_LVG_AREA": "sqft"})
        .round(1)
        .to_dict(orient="records")
    )

    return {
        "predicted": round(predicted),
        "confidence_low": round(max(0, predicted - rmse)),
        "confidence_high": round(predicted + rmse),
        "ocean_dist_m": round(ocean_dist),
        "cntr_dist_m": round(cntr_dist),
        "is_coastal": ocean_dist < 1000,
        "comparables": comps,
    }


# ── GET /api/neighbourhood-stats ─────────────────────────────────────────────
@app.get("/api/neighbourhood-stats")
def neighbourhood_stats():
    df: pd.DataFrame = _state["df"]
    result = []
    for nbhd, grp in df.groupby("neighbourhood"):
        result.append({
            "neighbourhood": nbhd,
            "median_price": float(grp["SALE_PRC"].median()),
            "mean_price": float(grp["SALE_PRC"].mean()),
            "count": int(len(grp)),
            "centroid_lat": float(grp["LATITUDE"].mean()),
            "centroid_lon": float(grp["LONGITUDE"].mean()),
            "bounds": [
                [float(grp["LATITUDE"].min()), float(grp["LONGITUDE"].min())],
                [float(grp["LATITUDE"].max()), float(grp["LONGITUDE"].max())],
            ],
        })
    return sorted(result, key=lambda x: x["median_price"], reverse=True)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "properties": len(_state.get("df", []))}

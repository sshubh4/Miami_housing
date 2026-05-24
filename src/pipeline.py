"""
Data pipeline for Miami Housing Price Predictor.

Handles ingestion, validation, cleaning, and feature engineering.
Designed to be reusable and testable — no global state, all pure functions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {
    "SALE_PRC",     # target: sale price in USD
    "LND_SQFOOT",   # land area sq ft
    "TOT_LVG_AREA", # total living area sq ft
    "SPEC_FEAT_VAL",# special features value
    "RAIL_DIST",    # distance to nearest rail station
    "OCEAN_DIST",   # distance to ocean
    "WATER_DIST",   # distance to water body
    "CNTR_DIST",    # distance to Miami CBD
    "SUBCNTR_DI",   # distance to nearest sub-centre
    "HWY_DIST",     # distance to nearest highway
    "age",          # property age
    "avno60plus",   # aircraft noise >60 decibels (0/1)
    "month_sold",   # month of sale (1–12)
    "LATITUDE",
    "LONGITUDE",
    "structure_quality", # 1–5 rating
}

PRICE_FLOOR = 10_000      # sanity: nothing sells for < $10k
PRICE_CEILING = 10_000_000  # remove extreme outliers (>$10M)


# ── Load ───────────────────────────────────────────────────────────────────────
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load CSV and drop the non-feature parcel ID column."""
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    df = df.drop(columns=["PARCELNO"], errors="ignore")
    return df


# ── Validate ──────────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    log.info("Schema OK — all %d required columns present", len(REQUIRED_COLS))


def validate_geo_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with coordinates outside Miami-Dade County bbox."""
    lat_ok = df["LATITUDE"].between(25.1, 26.0)
    lon_ok = df["LONGITUDE"].between(-80.9, -80.0)
    bad = (~lat_ok | ~lon_ok).sum()
    if bad:
        log.warning("Dropping %d rows outside Miami-Dade bbox", bad)
    return df[lat_ok & lon_ok].copy()


def validate_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with implausible sale prices."""
    ok = df["SALE_PRC"].between(PRICE_FLOOR, PRICE_CEILING)
    dropped = (~ok).sum()
    if dropped:
        log.warning("Dropping %d rows with price outside [%s, %s]",
                    dropped, f"${PRICE_FLOOR:,}", f"${PRICE_CEILING:,}")
    return df[ok].copy()


# ── Clean ─────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls and floor negative distance values at 0."""
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLS))
    log.info("Dropped %d rows with nulls (%d remain)", before - len(df), len(df))

    dist_cols = ["RAIL_DIST", "OCEAN_DIST", "WATER_DIST",
                 "CNTR_DIST", "SUBCNTR_DI", "HWY_DIST"]
    for col in dist_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that improve model signal.
    All transformations are documented with their rationale.
    """
    df = df.copy()

    # Price per sq ft — useful as a derived target for choropleth
    df["price_per_sqft"] = df["SALE_PRC"] / df["TOT_LVG_AREA"].replace(0, np.nan)

    # Log-distance features: diminishing returns on distance effects
    for col in ["OCEAN_DIST", "CNTR_DIST", "WATER_DIST"]:
        if col in df.columns:
            df[f"log_{col.lower()}"] = np.log1p(df[col])

    # Land-to-living ratio: high ratio = more land, typically higher value
    df["land_to_living"] = df["LND_SQFOOT"] / df["TOT_LVG_AREA"].replace(0, np.nan)

    # Coastal premium bin: properties within 1km of ocean command a premium
    df["is_coastal"] = (df["OCEAN_DIST"] < 1000).astype(int)

    # CBD accessibility score: closer to centre = higher demand
    df["cbd_access"] = 1 / (1 + df["CNTR_DIST"] / 1000)

    # Season of sale: Q1/Q2 = Miami peak season
    df["peak_season"] = df["month_sold"].isin([1, 2, 3, 4, 5]).astype(int)

    # Noise penalty: aircraft noise significantly suppresses price
    df["noise_penalty"] = df["avno60plus"] * df["OCEAN_DIST"]

    log.info("Feature engineering complete — %d columns", df.shape[1])
    return df.fillna(df.median(numeric_only=True))


# ── Neighbourhood enrichment (optional) ───────────────────────────────────────
def assign_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a rough neighbourhood label from lat/lon.
    These boundaries are approximate; replace with official shapefiles
    for production use (Miami-Dade Open Data portal has GeoJSON).
    """
    def _label(row):
        lat, lon = row["LATITUDE"], row["LONGITUDE"]
        if lat > 25.77 and lon > -80.2:
            return "Miami Beach"
        elif lat > 25.78 and lon < -80.3:
            return "Hialeah"
        elif lat < 25.6:
            return "Homestead / South Dade"
        elif lon < -80.5:
            return "West Dade"
        elif lat > 25.7 and lat < 25.77:
            return "Coral Gables / South Miami"
        else:
            return "Miami Core"

    df["neighbourhood"] = df.apply(_label, axis=1)
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(csv_path: str | Path) -> pd.DataFrame:
    """
    End-to-end pipeline: load → validate → clean → engineer → return.
    Returns a DataFrame ready for model training and map rendering.
    """
    df = load_raw(csv_path)
    validate_schema(df)
    df = validate_geo_bounds(df)
    df = validate_price_range(df)
    df = clean(df)
    df = engineer_features(df)
    df = assign_neighbourhood(df)
    log.info("Pipeline complete. Final shape: %s", df.shape)
    return df

"""
Tests for the Miami Housing pipeline and model.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    clean,
    engineer_features,
    validate_geo_bounds,
    validate_price_range,
    assign_neighbourhood,
    REQUIRED_COLS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal valid Miami-Dade DataFrame for testing."""
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "SALE_PRC":      rng.uniform(100_000, 800_000, n),
        "LND_SQFOOT":    rng.uniform(2000, 20000, n),
        "TOT_LVG_AREA":  rng.uniform(800, 5000, n),
        "SPEC_FEAT_VAL": rng.uniform(0, 50000, n),
        "RAIL_DIST":     rng.uniform(500, 20000, n),
        "OCEAN_DIST":    rng.uniform(200, 25000, n),
        "WATER_DIST":    rng.uniform(100, 15000, n),
        "CNTR_DIST":     rng.uniform(1000, 40000, n),
        "SUBCNTR_DI":    rng.uniform(500, 20000, n),
        "HWY_DIST":      rng.uniform(100, 10000, n),
        "age":           rng.integers(0, 70, n),
        "avno60plus":    rng.integers(0, 2, n),
        "month_sold":    rng.integers(1, 13, n),
        "structure_quality": rng.integers(1, 6, n),
        "LATITUDE":      rng.uniform(25.2, 25.9, n),
        "LONGITUDE":     rng.uniform(-80.85, -80.1, n),
    })


# ── Pipeline tests ─────────────────────────────────────────────────────────────
class TestGeoValidation:
    def test_drops_out_of_bounds_rows(self, sample_df):
        sample_df.loc[0, "LATITUDE"] = 30.0   # way north of Miami
        sample_df.loc[1, "LONGITUDE"] = -70.0  # way east
        result = validate_geo_bounds(sample_df)
        assert len(result) == len(sample_df) - 2

    def test_keeps_valid_rows(self, sample_df):
        result = validate_geo_bounds(sample_df)
        assert len(result) == len(sample_df)

    def test_lat_bounds(self, sample_df):
        result = validate_geo_bounds(sample_df)
        assert result["LATITUDE"].between(25.1, 26.0).all()


class TestPriceValidation:
    def test_drops_cheap_properties(self, sample_df):
        sample_df.loc[0, "SALE_PRC"] = 500   # implausibly cheap
        result = validate_price_range(sample_df)
        assert len(result) == len(sample_df) - 1

    def test_drops_luxury_outliers(self, sample_df):
        sample_df.loc[0, "SALE_PRC"] = 50_000_000  # way over ceiling
        result = validate_price_range(sample_df)
        assert len(result) == len(sample_df) - 1

    def test_keeps_normal_prices(self, sample_df):
        result = validate_price_range(sample_df)
        assert len(result) == len(sample_df)


class TestCleaning:
    def test_drops_null_rows(self, sample_df):
        sample_df.loc[0, "SALE_PRC"] = np.nan
        result = clean(sample_df)
        assert len(result) == len(sample_df) - 1

    def test_clips_negative_distances(self, sample_df):
        sample_df.loc[0, "OCEAN_DIST"] = -500
        result = clean(sample_df)
        assert (result["OCEAN_DIST"] >= 0).all()


class TestFeatureEngineering:
    def test_creates_price_per_sqft(self, sample_df):
        result = engineer_features(sample_df)
        assert "price_per_sqft" in result.columns

    def test_price_per_sqft_is_positive(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["price_per_sqft"].dropna() > 0).all()

    def test_creates_log_features(self, sample_df):
        result = engineer_features(sample_df)
        assert "log_ocean_dist" in result.columns
        assert "log_cntr_dist" in result.columns

    def test_is_coastal_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["is_coastal"].unique()).issubset({0, 1})

    def test_cbd_access_bounded(self, sample_df):
        result = engineer_features(sample_df)
        assert result["cbd_access"].between(0, 1).all()

    def test_no_nulls_after_engineering(self, sample_df):
        result = engineer_features(sample_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0


class TestNeighbourhoodAssignment:
    def test_all_rows_assigned(self, sample_df):
        result = assign_neighbourhood(sample_df)
        assert result["neighbourhood"].notna().all()

    def test_known_labels(self, sample_df):
        result = assign_neighbourhood(sample_df)
        valid = {"Miami Beach", "Hialeah", "Homestead / South Dade",
                 "West Dade", "Coral Gables / South Miami", "Miami Core"}
        assert set(result["neighbourhood"].unique()).issubset(valid)

    def test_miami_beach_lat_lon(self):
        row = pd.DataFrame([{
            "LATITUDE": 25.80, "LONGITUDE": -80.13
        }])
        from src.pipeline import assign_neighbourhood as an
        result = an(row)
        assert result["neighbourhood"].iloc[0] == "Miami Beach"

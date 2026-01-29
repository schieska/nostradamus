"""Tests for the forecasting engine."""

import numpy as np
import pandas as pd
import pytest


class TestCadenceDetection:
    """Tests for automatic cadence detection."""
    
    def test_detect_hourly_cadence(self, sample_dataframe):
        """Test detection of hourly cadence."""
        from nostradamus.cadence import detect_cadence
        
        cadence, confidence = detect_cadence(sample_dataframe)
        
        assert cadence == 3600  # 1 hour in seconds
        assert confidence > 0.8  # High confidence for regular data
    
    def test_detect_cadence_with_gaps(self, sample_dataframe):
        """Test cadence detection with some missing data."""
        from nostradamus.cadence import detect_cadence
        
        # Remove some random rows to simulate gaps
        df = sample_dataframe.drop(sample_dataframe.index[::7])  # Remove every 7th row
        
        cadence, confidence = detect_cadence(df)
        
        # Should still detect hourly cadence
        assert cadence == 3600
        # But with lower confidence
        assert confidence < 0.95
    
    def test_detect_5min_cadence(self):
        """Test detection of 5-minute cadence."""
        from nostradamus.cadence import detect_cadence
        
        # Create 5-minute data
        dates = pd.date_range(start="2026-01-01", periods=500, freq="5min")
        df = pd.DataFrame({"value": np.random.randn(len(dates))}, index=dates)
        
        cadence, confidence = detect_cadence(df)
        
        assert cadence == 300  # 5 minutes in seconds
    
    def test_resample_to_cadence(self, sample_dataframe):
        """Test resampling to a fixed cadence."""
        from nostradamus.cadence import resample_to_cadence
        
        # Resample hourly data to 2-hour cadence
        resampled = resample_to_cadence(sample_dataframe, 7200)  # 2 hours
        
        assert len(resampled) < len(sample_dataframe)
        assert len(resampled) == len(sample_dataframe) // 2


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_create_features(self, sample_dataframe, temp_storage):
        """Test feature creation."""
        from nostradamus.engine import ForecastEngine
        
        # Create engine with mock token
        engine = ForecastEngine(supervisor_token="test", storage=temp_storage)
        
        # Create features
        X, y, feature_names = engine._create_features(
            sample_dataframe,
            {},  # No supporting entities
            horizon=24
        )
        
        assert len(X) > 0
        assert len(y) == len(X)
        assert len(feature_names) > 0
        
        # Check expected features exist
        assert "target_lag_1" in feature_names
        assert "hour" in feature_names
        assert "day_of_week" in feature_names
    
    def test_create_features_with_supporting(
        self, sample_dataframe, sample_binary_dataframe, temp_storage
    ):
        """Test feature creation with supporting entities."""
        from nostradamus.engine import ForecastEngine
        
        engine = ForecastEngine(supervisor_token="test", storage=temp_storage)
        
        supporting_dfs = {
            "binary_sensor.window": sample_binary_dataframe
        }
        
        X, y, feature_names = engine._create_features(
            sample_dataframe,
            supporting_dfs,
            horizon=24
        )
        
        # Check supporting entity features exist
        assert any("binary_sensor_window" in f for f in feature_names)


class TestModelTraining:
    """Tests for model training."""
    
    def test_train_models(self, sample_dataframe, temp_storage):
        """Test LightGBM model training."""
        from nostradamus.engine import ForecastEngine
        
        engine = ForecastEngine(supervisor_token="test", storage=temp_storage)
        
        X, y, _ = engine._create_features(sample_dataframe, {}, horizon=24)
        
        models = engine._train_models(X, y)
        
        assert "lower" in models
        assert "median" in models
        assert "upper" in models
        
        # Test predictions
        pred = models["median"].predict(X[:1])
        assert len(pred) == 1
    
    def test_quantile_ordering(self, sample_dataframe, temp_storage):
        """Test that quantile predictions are properly ordered."""
        from nostradamus.engine import ForecastEngine
        
        engine = ForecastEngine(supervisor_token="test", storage=temp_storage)
        
        X, y, _ = engine._create_features(sample_dataframe, {}, horizon=24)
        models = engine._train_models(X, y)
        
        # Predict on test data
        lower = models["lower"].predict(X)
        median = models["median"].predict(X)
        upper = models["upper"].predict(X)
        
        # On average, lower < median < upper
        # (individual points might violate this due to quantile crossing)
        assert np.mean(lower) < np.mean(median)
        assert np.mean(median) < np.mean(upper)


class TestStorage:
    """Tests for forecast storage."""
    
    def test_save_and_load(self, temp_storage):
        """Test saving and loading forecasts."""
        forecast_data = {
            "id": "test_forecast",
            "name": "Test Forecast",
            "target_entity": "sensor.test",
            "horizon": 24
        }
        
        temp_storage.save("test_forecast", forecast_data)
        
        loaded = temp_storage.get("test_forecast")
        
        assert loaded is not None
        assert loaded["id"] == "test_forecast"
        assert loaded["name"] == "Test Forecast"
        assert "updated_at" in loaded
    
    def test_list_all(self, temp_storage):
        """Test listing all forecasts."""
        # Save a few forecasts
        for i in range(3):
            temp_storage.save(f"forecast_{i}", {
                "id": f"forecast_{i}",
                "name": f"Forecast {i}",
                "target_entity": f"sensor.test_{i}",
                "horizon": 24
            })
        
        forecasts = temp_storage.list_all()
        
        assert len(forecasts) == 3
    
    def test_delete(self, temp_storage):
        """Test deleting a forecast."""
        temp_storage.save("to_delete", {"id": "to_delete", "name": "Delete Me"})
        
        assert temp_storage.exists("to_delete")
        
        temp_storage.delete("to_delete")
        
        assert not temp_storage.exists("to_delete")

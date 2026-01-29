"""LightGBM-based forecasting engine."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from .cadence import detect_cadence, resample_to_cadence, get_required_history_days
from .data_fetcher import HADataFetcher
from .storage import ForecastStorage

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Core forecasting engine using LightGBM."""
    
    def __init__(self, supervisor_token: str, storage: ForecastStorage):
        """
        Initialize the forecast engine.
        
        Args:
            supervisor_token: Token for HA API access
            storage: Storage instance for persisting forecasts
        """
        self.data_fetcher = HADataFetcher(token=supervisor_token)
        self.storage = storage
        self.models: Dict[str, Dict[str, lgb.Booster]] = {}
    
    def get_available_entities(self) -> List[Dict[str, Any]]:
        """Get list of available numeric entities from HA."""
        return self.data_fetcher.get_numeric_entities()
    
    def create_forecast(
        self,
        forecast_id: str,
        name: str,
        target_entity: str,
        supporting_entities: List[str],
        horizon: int
    ) -> Dict[str, Any]:
        """
        Create a new forecast device and train initial model.
        
        Args:
            forecast_id: Unique identifier for this forecast
            name: Human-readable name
            target_entity: Entity ID to predict
            supporting_entities: Additional entity IDs to use as features
            horizon: Number of steps to forecast
            
        Returns:
            Initial forecast results
        """
        logger.info(f"Creating forecast '{name}' for {target_entity} with horizon {horizon}")
        
        # Fetch initial history to detect cadence
        all_entities = [target_entity] + supporting_entities
        history = self.data_fetcher.get_history(all_entities, days_back=7)
        
        if target_entity not in history:
            raise ValueError(f"No history found for target entity: {target_entity}")
        
        # Detect cadence from target entity
        target_df = history[target_entity]
        cadence_seconds, cadence_confidence = detect_cadence(target_df)
        
        # Store configuration
        config = {
            "id": forecast_id,
            "name": name,
            "target_entity": target_entity,
            "supporting_entities": supporting_entities,
            "horizon": horizon,
            "cadence_seconds": cadence_seconds,
            "cadence_confidence": cadence_confidence,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_trained": None,
            "status": "training"
        }
        self.storage.save(forecast_id, config)
        
        # Start training in background
        import threading
        thread = threading.Thread(
            target=self._train_and_forecast_background,
            args=(forecast_id, config, history)
        )
        thread.start()
        
        return config

    def _train_and_forecast_background(
        self,
        forecast_id: str,
        config: Dict[str, Any],
        history: Dict[str, pd.DataFrame]
    ) -> None:
        """Background wrapper for training."""
        try:
            self._train_and_forecast(forecast_id, config, history)
        except Exception as e:
            logger.exception(f"Background training failed for {forecast_id}")
            config["status"] = "error"
            config["error"] = str(e)
            self.storage.save(forecast_id, config)
    
    def retrain(self, forecast_id: str) -> Dict[str, Any]:
        """
        Retrain an existing forecast with fresh data.
        
        Args:
            forecast_id: ID of forecast to retrain
            
        Returns:
            Updated forecast results
        """
        config = self.storage.get(forecast_id)
        if config is None:
            raise ValueError(f"Forecast not found: {forecast_id}")
        
        logger.info(f"Retraining forecast {forecast_id}")
        
        # Fetch fresh history
        all_entities = [config["target_entity"]] + config.get("supporting_entities", [])
        days_needed = get_required_history_days(config["horizon"], config["cadence_seconds"])
        history = self.data_fetcher.get_history(all_entities, days_back=days_needed)
        
        if config["target_entity"] not in history:
            logger.error(f"No history found for target entity: {config['target_entity']}")
            config["status"] = "error"
            config["error"] = "No history data available"
            self.storage.save(forecast_id, config)
            return config
        
        # Retrain and generate forecast
        result = self._train_and_forecast(forecast_id, config, history)
        
        return result
    
    def _train_and_forecast(
        self,
        forecast_id: str,
        config: Dict[str, Any],
        history: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Train model and generate forecast.
        
        Args:
            forecast_id: Forecast ID
            config: Forecast configuration
            history: Historical data for all entities
            
        Returns:
            Updated config with forecast results
        """
        target_entity = config["target_entity"]
        supporting_entities = config.get("supporting_entities", [])
        horizon = config["horizon"]
        cadence_seconds = config["cadence_seconds"]
        
        # Prepare data
        target_df = history[target_entity]
        target_resampled = resample_to_cadence(target_df, cadence_seconds)
        
        if len(target_resampled) < horizon * 2:
            raise ValueError(
                f"Insufficient data: need at least {horizon * 2} points, "
                f"got {len(target_resampled)}"
            )
        
        # Prepare supporting features
        supporting_dfs = {}
        for entity_id in supporting_entities:
            if entity_id in history:
                df = history[entity_id]
                supporting_dfs[entity_id] = resample_to_cadence(df, cadence_seconds)
        
        # Create feature matrix
        X, y, feature_names = self._create_features(
            target_resampled,
            supporting_dfs,
            horizon
        )
        
        if len(X) < 50:
            raise ValueError(f"Insufficient training samples: {len(X)} (need at least 50)")
        
        # Train models for quantile regression (lower, median, upper)
        models = self._train_models(X, y)
        self.models[forecast_id] = models
        
        # Save models
        model_path = self.storage.get_model_path(forecast_id)
        models["median"].save_model(str(model_path))
        
        # Generate forecast
        forecast_result = self._generate_forecast(
            target_resampled,
            supporting_dfs,
            models,
            horizon,
            cadence_seconds
        )
        
        # Calculate confidence score based on prediction interval width
        forecast_values = forecast_result["forecast"]
        lower_bound = forecast_result["lower_bound"]
        upper_bound = forecast_result["upper_bound"]
        
        mean_width = np.mean([u - l for u, l in zip(upper_bound, lower_bound)])
        mean_value = np.mean(np.abs(forecast_values))
        
        if mean_value > 0:
            relative_width = mean_width / mean_value
            confidence_score = max(0, min(100, int((1 - relative_width) * 100)))
        else:
            confidence_score = 50
        
        # Update config with results
        config.update({
            "status": "ok",
            "last_trained": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "next_value": forecast_values[0] if forecast_values else None,
            "forecast": forecast_values,
            "timestamps": forecast_result["timestamps"],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_score": confidence_score,
            "training_samples": len(X),
            "feature_names": feature_names
        })
        
        self.storage.save(forecast_id, config)
        
        logger.info(
            f"Trained {forecast_id}: {len(X)} samples, "
            f"confidence {confidence_score}%"
        )
        
        return config
    
    def _create_features(
        self,
        target_df: pd.DataFrame,
        supporting_dfs: Dict[str, pd.DataFrame],
        horizon: int
    ) -> tuple:
        """
        Create feature matrix for training.
        
        Features include:
        - Lagged values of target (1, 2, 3, ..., horizon steps)
        - Time features (hour, day of week, month)
        - Supporting entity values (lagged)
        
        Args:
            target_df: Resampled target data
            supporting_dfs: Resampled supporting entity data
            horizon: Forecast horizon
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = target_df.copy()
        df.columns = ["target"]
        feature_names = []
        
        # Create lagged features for target
        lags = [1, 2, 3, 6, 12, 24] + [horizon]
        lags = sorted(set([l for l in lags if l <= len(df) // 2]))
        
        for lag in lags:
            df[f"target_lag_{lag}"] = df["target"].shift(lag)
            feature_names.append(f"target_lag_{lag}")
        
        # Rolling statistics
        for window in [6, 12, 24]:
            if window <= len(df) // 4:
                df[f"target_rolling_mean_{window}"] = df["target"].shift(1).rolling(window).mean()
                df[f"target_rolling_std_{window}"] = df["target"].shift(1).rolling(window).std()
                feature_names.extend([
                    f"target_rolling_mean_{window}",
                    f"target_rolling_std_{window}"
                ])
        
        # Time features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        feature_names.extend(["hour", "day_of_week", "month", "is_weekend"])
        
        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        feature_names.extend(["hour_sin", "hour_cos"])
        
        # Supporting entity features
        for entity_id, entity_df in supporting_dfs.items():
            entity_df = entity_df.reindex(df.index, method="ffill")
            safe_name = entity_id.replace(".", "_")
            
            df[f"{safe_name}_value"] = entity_df["value"]
            df[f"{safe_name}_lag_1"] = entity_df["value"].shift(1)
            feature_names.extend([f"{safe_name}_value", f"{safe_name}_lag_1"])
        
        # Target is the value 'horizon' steps in the future
        df["y"] = df["target"].shift(-int(horizon))
        
        # Drop rows with NaN
        df = df.dropna()
        
        X = df[feature_names].values
        y = df["y"].values
        
        return X, y, feature_names
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, lgb.Booster]:
        """
        Train LightGBM models for quantile regression.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with 'lower', 'median', 'upper' models
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Common parameters
        base_params = {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": 1
        }
        
        models = {}
        
        # Train for each quantile
        for name, alpha in [("lower", 0.1), ("median", 0.5), ("upper", 0.9)]:
            params = base_params.copy()
            params["alpha"] = alpha
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            models[name] = model
        
        return models
    
    def _generate_forecast(
        self,
        target_df: pd.DataFrame,
        supporting_dfs: Dict[str, pd.DataFrame],
        models: Dict[str, lgb.Booster],
        horizon: int,
        cadence_seconds: int
    ) -> Dict[str, Any]:
        """
        Generate forecast for the next 'horizon' steps.
        
        This uses an iterative approach where each predicted value is used
        as input for predicting the next step.
        
        Args:
            target_df: Historical target data
            supporting_dfs: Historical supporting entity data
            models: Trained LightGBM models
            horizon: Number of steps to forecast
            cadence_seconds: Cadence in seconds
            
        Returns:
            Dictionary with forecast, timestamps, lower_bound, upper_bound
        """
        forecast = []
        lower_bound = []
        upper_bound = []
        timestamps = []
        
        # Start from the last known timestamp
        last_timestamp = target_df.index[-1]
        
        # Create a working copy of the data
        working_df = target_df.copy()
        working_df.columns = ["target"]
        
        for step in range(horizon):
            # Create features for this step
            X_step = self._create_step_features(working_df, supporting_dfs, step)
            
            # Predict
            pred_median = float(models["median"].predict(X_step)[0])
            pred_lower = float(models["lower"].predict(X_step)[0])
            pred_upper = float(models["upper"].predict(X_step)[0])
            
            # Calculate timestamp
            next_timestamp = last_timestamp + timedelta(seconds=cadence_seconds * (step + 1))
            
            forecast.append(round(pred_median, 3))
            lower_bound.append(round(pred_lower, 3))
            upper_bound.append(round(pred_upper, 3))
            timestamps.append(next_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"))
            
            # Add prediction to working data for next iteration
            working_df.loc[next_timestamp] = pred_median
        
        return {
            "forecast": forecast,
            "timestamps": timestamps,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def _create_step_features(
        self,
        working_df: pd.DataFrame,
        supporting_dfs: Dict[str, pd.DataFrame],
        step: int
    ) -> np.ndarray:
        """Create features for a single forecast step."""
        # Get the last row with enough history
        target_series = working_df["target"]
        last_idx = working_df.index[-1]
        
        features = []
        
        # Lagged features
        lags = [1, 2, 3, 6, 12, 24]
        for lag in lags:
            if lag <= len(target_series):
                features.append(target_series.iloc[-lag])
            else:
                features.append(target_series.iloc[0])
        
        # Rolling statistics (use available data)
        for window in [6, 12, 24]:
            if window <= len(target_series):
                rolling_data = target_series.tail(window)
                features.append(rolling_data.mean())
                features.append(rolling_data.std())
            else:
                features.extend([target_series.mean(), target_series.std()])
        
        # Time features for the forecast timestamp
        forecast_time = last_idx + timedelta(seconds=step * 60)  # Approximate
        features.append(forecast_time.hour)
        features.append(forecast_time.dayofweek)
        features.append(forecast_time.month)
        features.append(1 if forecast_time.dayofweek >= 5 else 0)
        features.append(np.sin(2 * np.pi * forecast_time.hour / 24))
        features.append(np.cos(2 * np.pi * forecast_time.hour / 24))
        
        # Supporting entity features
        for entity_id, entity_df in supporting_dfs.items():
            if len(entity_df) > 0:
                last_value = entity_df["value"].iloc[-1]
                prev_value = entity_df["value"].iloc[-2] if len(entity_df) > 1 else last_value
            else:
                last_value = 0
                prev_value = 0
            
            features.append(last_value)
            features.append(prev_value)
        
        return np.array([features])

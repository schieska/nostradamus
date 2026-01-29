"""Forecast storage management."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ForecastStorage:
    """Manages persistent storage of forecast configurations and results."""
    
    def __init__(self, data_path: str):
        """Initialize storage with data directory path."""
        self.data_path = Path(data_path)
        self.forecasts_dir = self.data_path / "forecasts"
        self.models_dir = self.data_path / "models"
        
        # Ensure directories exist
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage initialized at {self.data_path}")
    
    def _forecast_path(self, forecast_id: str) -> Path:
        """Get path to forecast JSON file."""
        return self.forecasts_dir / f"{forecast_id}.json"
    
    def _model_path(self, forecast_id: str) -> Path:
        """Get path to model file."""
        return self.models_dir / f"{forecast_id}.lgb"
    
    def exists(self, forecast_id: str) -> bool:
        """Check if a forecast exists."""
        return self._forecast_path(forecast_id).exists()
    
    def save(self, forecast_id: str, data: Dict[str, Any]) -> None:
        """Save forecast data to storage."""
        data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        path = self._forecast_path(forecast_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved forecast {forecast_id}")
    
    def get(self, forecast_id: str) -> Optional[Dict[str, Any]]:
        """Load forecast data from storage."""
        path = self._forecast_path(forecast_id)
        if not path.exists():
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def delete(self, forecast_id: str) -> None:
        """Delete a forecast and its model."""
        forecast_path = self._forecast_path(forecast_id)
        model_path = self._model_path(forecast_id)
        
        if forecast_path.exists():
            forecast_path.unlink()
        if model_path.exists():
            model_path.unlink()
        
        logger.info(f"Deleted forecast {forecast_id}")
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all forecast configurations."""
        forecasts = []
        for path in self.forecasts_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    # Return summary only
                    forecasts.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "target_entity": data.get("target_entity"),
                        "horizon": data.get("horizon"),
                        "cadence_seconds": data.get("cadence_seconds"),
                        "last_trained": data.get("last_trained"),
                        "confidence_score": data.get("confidence_score")
                    })
            except Exception as e:
                logger.error(f"Failed to load forecast {path}: {e}")
        
        return forecasts
    
    def get_model_path(self, forecast_id: str) -> Path:
        """Get the path where a model should be stored."""
        return self._model_path(forecast_id)
    
    def list_forecast_ids(self) -> List[str]:
        """List all forecast IDs."""
        return [p.stem for p in self.forecasts_dir.glob("*.json")]

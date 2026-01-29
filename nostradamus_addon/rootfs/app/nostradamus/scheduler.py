"""Scheduler for periodic forecast retraining."""

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from .engine import ForecastEngine
    from .storage import ForecastStorage

logger = logging.getLogger(__name__)


class ForecastScheduler:
    """Manages scheduled retraining of forecasts."""
    
    def __init__(self, engine: "ForecastEngine", storage: "ForecastStorage"):
        """
        Initialize the scheduler.
        
        Args:
            engine: Forecast engine instance
            storage: Storage instance
        """
        self.engine = engine
        self.storage = storage
        self.scheduler = BackgroundScheduler()
        
        # Add main job that retrains all forecasts every hour
        self.scheduler.add_job(
            self._retrain_all,
            trigger=IntervalTrigger(hours=1),
            id="retrain_all",
            name="Retrain all forecasts",
            replace_existing=True
        )
    
    def start(self) -> None:
        """Start the scheduler."""
        self.scheduler.start()
        logger.info("Forecast scheduler started")
        
        # Run initial retrain for any existing forecasts
        self._retrain_all()
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=True)
        logger.info("Forecast scheduler stopped")
    
    def remove_job(self, forecast_id: str) -> None:
        """Remove a job for a specific forecast (if exists)."""
        job_id = f"retrain_{forecast_id}"
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass  # Job might not exist
    
    def _retrain_all(self) -> None:
        """Retrain all existing forecasts."""
        forecast_ids = self.storage.list_forecast_ids()
        
        if not forecast_ids:
            logger.debug("No forecasts to retrain")
            return
        
        logger.info(f"Starting scheduled retrain for {len(forecast_ids)} forecasts")
        
        for forecast_id in forecast_ids:
            try:
                self.engine.retrain(forecast_id)
            except Exception as e:
                logger.exception(f"Failed to retrain {forecast_id}: {e}")
        
        logger.info("Scheduled retrain complete")

"""Flask application factory and REST API endpoints."""

import logging
import os
from flask import Flask, jsonify, request

from .engine import ForecastEngine
from .scheduler import ForecastScheduler
from .storage import ForecastStorage

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config["DATA_PATH"] = os.environ.get("DATA_PATH", "/data")
    app.config["SUPERVISOR_TOKEN"] = os.environ.get("SUPERVISOR_TOKEN", "")
    
    # Initialize components
    storage = ForecastStorage(app.config["DATA_PATH"])
    engine = ForecastEngine(
        supervisor_token=app.config["SUPERVISOR_TOKEN"],
        storage=storage
    )
    scheduler = ForecastScheduler(engine, storage)
    
    # Store in app context
    app.engine = engine
    app.storage = storage
    app.scheduler = scheduler
    
    # Start scheduler
    scheduler.start()
    
    # Register routes
    register_routes(app)
    
    logger.info("Nostradamus API initialized")
    return app


def register_routes(app: Flask) -> None:
    """Register all API routes."""
    
    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "version": "0.1.0"})
    
    @app.route("/api/forecasts", methods=["GET"])
    def list_forecasts():
        """List all configured forecast devices."""
        forecasts = app.storage.list_all()
        return jsonify({"forecasts": forecasts})
    
    @app.route("/api/forecasts", methods=["POST"])
    def create_forecast():
        """Create a new forecast device."""
        data = request.get_json()
        
        # Validate required fields
        required = ["id", "name", "target_entity", "horizon"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        
        try:
            # Create forecast device
            result = app.engine.create_forecast(
                forecast_id=data["id"],
                name=data["name"],
                target_entity=data["target_entity"],
                supporting_entities=data.get("supporting_entities", []),
                horizon=data["horizon"]
            )
            return jsonify(result), 201
        except Exception as e:
            logger.exception("Failed to create forecast")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/forecasts/<forecast_id>", methods=["GET"])
    def get_forecast(forecast_id: str):
        """Get forecast results for a device."""
        forecast = app.storage.get(forecast_id)
        if forecast is None:
            return jsonify({"error": "Forecast not found"}), 404
        return jsonify(forecast)
    
    @app.route("/api/forecasts/<forecast_id>", methods=["DELETE"])
    def delete_forecast(forecast_id: str):
        """Delete a forecast device."""
        if not app.storage.exists(forecast_id):
            return jsonify({"error": "Forecast not found"}), 404
        
        app.storage.delete(forecast_id)
        app.scheduler.remove_job(forecast_id)
        return jsonify({"status": "deleted"})
    
    @app.route("/api/forecasts/<forecast_id>/retrain", methods=["POST"])
    def retrain_forecast(forecast_id: str):
        """Trigger manual retrain for a forecast device."""
        if not app.storage.exists(forecast_id):
            return jsonify({"error": "Forecast not found"}), 404
        
        try:
            result = app.engine.retrain(forecast_id)
            return jsonify(result)
        except Exception as e:
            logger.exception("Failed to retrain forecast")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/entities", methods=["GET"])
    def list_entities():
        """List available numeric entities from Home Assistant."""
        try:
            entities = app.engine.get_available_entities()
            return jsonify({"entities": entities})
        except Exception as e:
            logger.exception("Failed to list entities")
            return jsonify({"error": str(e)}), 500


# For development/testing without Gunicorn
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

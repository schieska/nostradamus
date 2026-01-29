"""Constants for the Nostradamus integration."""

DOMAIN = "nostradamus"

# Configuration keys
CONF_TARGET_ENTITY = "target_entity"
CONF_SUPPORTING_ENTITIES = "supporting_entities"
CONF_HORIZON = "horizon"
CONF_NAME = "name"
CONF_ADDON_HOST = "addon_host"

# Defaults
DEFAULT_ADDON_HOST = "http://local-nostradamus:5000"
DEFAULT_HORIZON = 24

# API endpoints
API_HEALTH = "/api/health"
API_FORECASTS = "/api/forecasts"
API_ENTITIES = "/api/entities"

# Update intervals
SCAN_INTERVAL_SECONDS = 300  # 5 minutes

# Sensor attributes
ATTR_FORECAST = "forecast"
ATTR_TIMESTAMPS = "timestamps"
ATTR_LOWER_BOUND = "lower_bound"
ATTR_UPPER_BOUND = "upper_bound"
ATTR_CONFIDENCE_SCORE = "confidence_score"
ATTR_LAST_TRAINED = "last_trained"
ATTR_CADENCE = "cadence_seconds"
ATTR_TARGET_ENTITY = "target_entity"
ATTR_SUPPORTING_ENTITIES = "supporting_entities"

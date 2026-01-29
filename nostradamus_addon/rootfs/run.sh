#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Nostradamus Forecasting Engine..."

# Set up environment
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export DATA_PATH="/data"

# Start the Flask application with Gunicorn
cd /app
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --log-level info \
    "nostradamus.main:create_app()"

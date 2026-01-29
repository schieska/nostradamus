# Nostradamus Add-on

ML-powered sensor forecasting for Home Assistant.

## About

This add-on provides the machine learning backend for Nostradamus. It:

- Fetches sensor history from Home Assistant
- Trains LightGBM models for forecasting
- Exposes a REST API for the integration to consume
- Automatically retrains models every hour

## Installation

1. Copy this folder to `/addons/nostradamus` on your HA machine
2. Go to **Settings → Add-ons → Add-on Store**
3. Click **⋮ → Check for updates**
4. Find Nostradamus and install

## Configuration

No configuration required. The add-on auto-configures using the Supervisor API.

## API

The add-on runs on port 5000 (internal only) and exposes:

- `GET /api/health` - Health check
- `GET /api/forecasts` - List forecasts
- `POST /api/forecasts` - Create forecast
- `GET /api/forecasts/{id}` - Get forecast
- `DELETE /api/forecasts/{id}` - Delete forecast
- `POST /api/forecasts/{id}/retrain` - Trigger retrain

## Requirements

- Home Assistant OS or Supervised installation
- At least 256MB RAM available
- Sensors with 7+ days of history

## Support

Report issues at: https://github.com/your-repo/nostradamus

# Nostradamus

**ML-powered sensor forecasting for Home Assistant**

"Pick a sensor, pick what influences it, see the future."

## Overview

Nostradamus lets Home Assistant users forecast any numeric sensor using historical data and user-selected supporting sensors. It exposes predictions as standard HA sensors that can be used in automations and dashboards.

## Architecture

Nostradamus consists of two parts:

1. **Add-on (ML Backend)** - Docker container running LightGBM for training and forecasting
2. **Integration (HA UX)** - Custom component providing config flow and sensor entities

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant                            │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │ Nostradamus     │     │ Nostradamus Add-on          │   │
│  │ Integration     │◄───►│ (Flask + LightGBM)          │   │
│  │                 │     │                             │   │
│  │ • Config Flow   │     │ • REST API                  │   │
│  │ • Sensors       │     │ • Forecast Engine           │   │
│  │ • Coordinator   │     │ • Hourly Retrain            │   │
│  └─────────────────┘     └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation via GitHub (Recommended)

Once you've pushed this code to a GitHub repository (e.g., `https://github.com/schieska/nostradamus`), you can install both parts directly:

### 1. Install the Add-on
1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Click the three dots (⋮) in the top right and select **Repositories**.
3. Add your GitHub URL: `https://github.com/schieska/nostradamus`.
4. Close the dialog. "Nostradamus" should now appear in the list (you may need to refresh).
5. Click **Nostradamus** and then **Install**.

### 2. Install the Integration (via HACS)
1. In Home Assistant, go to **HACS → Integrations**.
2. Click the three dots (⋮) in the top right and select **Custom repositories**.
3. Paste the GitHub URL: `https://github.com/schieska/nostradamus`.
4. Select **Integration** as the category and click **Add**.
5. Find "Nostradamus" in HACS and click **Download**.
6. Restart Home Assistant.

---

## Manual Installation (Local Development)

### 1. Install the Add-on
Copy the `nostradamus_addon` folder to your Home Assistant's `/addons/` directory:

```bash
# On your HA machine
cp -r nostradamus_addon /addons/nostradamus
```

Then in Home Assistant:
1. Go to **Settings → Add-ons → Add-on Store**
2. Click the menu (⋮) → **Check for updates**
3. Find Nostradamus and click **Install**
4. Start the add-on

### 2. Install the Integration

Copy the `custom_components/nostradamus` folder to your HA config:

```bash
cp -r custom_components/nostradamus /config/custom_components/
```

Restart Home Assistant.

### 3. Configure a Forecast

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for "Nostradamus"
3. Follow the config flow:
   - Verify add-on connection
   - Select target sensor to predict
   - Optionally select supporting sensors
   - Set forecast horizon (number of steps)

Two sensors will be created:
- `sensor.nostradamus_<name>` - The predicted value
- `sensor.nostradamus_<name>_confidence` - Confidence score (0-100%)

## Testing (Development)

### Setup

```powershell
# Install dependencies
pip install -r requirements-dev.txt

# Create .env file
cp .env.example .env
# Edit .env with your HA URL and long-lived access token
```

### Run Unit Tests

```powershell
python -m pytest tests/test_engine.py -v
```

### Interactive Testing

```powershell
# List available sensors
python tests/test_harness.py --list-entities

# Test forecast generation
python tests/test_harness.py --entity sensor.power_usage --horizon 24 --visualize
```

## Configuration

### Horizon

The horizon is specified in **steps**, where each step corresponds to the detected update frequency of your sensor:

- If your sensor updates every 5 minutes, horizon=24 means 2 hours ahead
- If your sensor updates hourly, horizon=24 means 24 hours ahead

The cadence is automatically detected from your sensor's history.

### Supporting Entities

Supporting entities can improve forecast accuracy by providing context:

- **For indoor temperature**: outdoor temperature, window sensors, HVAC state
- **For power usage**: time of day, occupancy sensors, weather
- **For solar production**: weather forecasts, cloud cover

## API Reference

The add-on exposes a REST API at `http://local-nostradamus:5000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/forecasts` | GET | List all forecasts |
| `/api/forecasts` | POST | Create forecast |
| `/api/forecasts/{id}` | GET | Get forecast |
| `/api/forecasts/{id}` | DELETE | Delete forecast |
| `/api/forecasts/{id}/retrain` | POST | Trigger retrain |

## How It Works

1. **Data Collection**: Fetches sensor history via HA API
2. **Cadence Detection**: Automatically detects sensor update frequency
3. **Feature Engineering**: Creates lag features, rolling stats, time features
4. **Model Training**: Trains LightGBM quantile regression models
5. **Forecasting**: Generates predictions with confidence bounds
6. **Retraining**: Automatically retrains every hour with fresh data

## Troubleshooting

### Add-on won't start
- Check logs: **Settings → Add-ons → Nostradamus → Log**
- Ensure sufficient memory (256MB recommended)

### No forecast data
- Verify the target sensor has at least 7 days of history
- Check that the sensor provides numeric values

### Low confidence score
- Add more supporting entities
- Ensure consistent sensor data (fewer gaps)
- Wait for more history to accumulate

## License

MIT

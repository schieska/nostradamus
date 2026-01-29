"""Data update coordinator for Nostradamus."""

import logging
from datetime import timedelta
from typing import Any, Dict, Optional

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    API_FORECASTS,
    CONF_ADDON_HOST,
    CONF_HORIZON,
    CONF_NAME,
    CONF_SUPPORTING_ENTITIES,
    CONF_TARGET_ENTITY,
    DEFAULT_ADDON_HOST,
    DOMAIN,
    SCAN_INTERVAL_SECONDS,
)

_LOGGER = logging.getLogger(__name__)


class NostradamusCoordinator(DataUpdateCoordinator[Dict[str, Any]]):
    """Coordinator to manage data updates from the Nostradamus add-on."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=SCAN_INTERVAL_SECONDS),
        )
        
        self.entry = entry
        self.addon_host = entry.data.get(CONF_ADDON_HOST, DEFAULT_ADDON_HOST)
        self.forecast_id = self._generate_forecast_id()
        self._forecast_created = False

    def _generate_forecast_id(self) -> str:
        """Generate a forecast ID from config entry."""
        target = self.entry.data.get(CONF_TARGET_ENTITY, "unknown")
        # Create a safe ID
        safe_id = target.replace(".", "_").replace(" ", "_").lower()
        return f"ha_{safe_id}"

    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch data from the add-on."""
        try:
            # Ensure forecast is created
            if not self._forecast_created:
                await self._create_forecast()
            
            # Get latest forecast
            return await self._get_forecast()
            
        except aiohttp.ClientError as err:
            raise UpdateFailed(f"Error communicating with add-on: {err}") from err
        except Exception as err:
            _LOGGER.exception("Unexpected error fetching forecast")
            raise UpdateFailed(f"Unexpected error: {err}") from err

    async def _create_forecast(self) -> None:
        """Create the forecast in the add-on if it doesn't exist."""
        url = f"{self.addon_host.rstrip('/')}{API_FORECASTS}"
        
        payload = {
            "id": self.forecast_id,
            "name": self.entry.data.get(CONF_NAME, "Forecast"),
            "target_entity": self.entry.data.get(CONF_TARGET_ENTITY),
            "supporting_entities": self.entry.data.get(CONF_SUPPORTING_ENTITIES, []),
            "horizon": self.entry.data.get(CONF_HORIZON, 24),
        }
        
        _LOGGER.info(f"Creating forecast: {payload}")
        
        async with aiohttp.ClientSession() as session:
            # First check if forecast exists
            get_url = f"{url}/{self.forecast_id}"
            async with session.get(get_url, timeout=10) as response:
                if response.status == 200:
                    _LOGGER.info(f"Forecast {self.forecast_id} already exists")
                    self._forecast_created = True
                    return
            
            # Create new forecast
            async with session.post(url, json=payload, timeout=120) as response:
                if response.status == 201:
                    _LOGGER.info(f"Forecast {self.forecast_id} created")
                    self._forecast_created = True
                elif response.status == 409:
                    _LOGGER.info(f"Forecast {self.forecast_id} already exists")
                    self._forecast_created = True
                else:
                    text = await response.text()
                    raise UpdateFailed(f"Failed to create forecast: {response.status} - {text}")

    async def _get_forecast(self) -> Dict[str, Any]:
        """Get the current forecast from the add-on."""
        url = f"{self.addon_host.rstrip('/')}{API_FORECASTS}/{self.forecast_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    # Forecast doesn't exist, recreate it
                    self._forecast_created = False
                    await self._create_forecast()
                    return await self._get_forecast()
                else:
                    text = await response.text()
                    raise UpdateFailed(f"Failed to get forecast: {response.status} - {text}")

    async def async_retrain(self) -> bool:
        """Trigger a manual retrain."""
        url = f"{self.addon_host.rstrip('/')}{API_FORECASTS}/{self.forecast_id}/retrain"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, timeout=120) as response:
                    if response.status == 200:
                        await self.async_request_refresh()
                        return True
                    else:
                        _LOGGER.error(f"Retrain failed: {response.status}")
                        return False
        except Exception as e:
            _LOGGER.exception(f"Retrain error: {e}")
            return False

"""Sensor platform for Nostradamus."""

import logging
from typing import Any, Dict, List, Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_CADENCE,
    ATTR_CONFIDENCE_SCORE,
    ATTR_FORECAST,
    ATTR_LAST_TRAINED,
    ATTR_LOWER_BOUND,
    ATTR_SUPPORTING_ENTITIES,
    ATTR_TARGET_ENTITY,
    ATTR_TIMESTAMPS,
    ATTR_UPPER_BOUND,
    CONF_NAME,
    CONF_TARGET_ENTITY,
    DOMAIN,
)
from .coordinator import NostradamusCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Nostradamus sensors from a config entry."""
    coordinator: NostradamusCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    
    entities = [
        NostradamusForecastSensor(coordinator, entry),
        NostradamusConfidenceSensor(coordinator, entry),
    ]
    
    async_add_entities(entities)


class NostradamusBaseSensor(CoordinatorEntity[NostradamusCoordinator], SensorEntity):
    """Base class for Nostradamus sensors."""

    def __init__(
        self,
        coordinator: NostradamusCoordinator,
        entry: ConfigEntry,
        sensor_type: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        
        self._entry = entry
        self._sensor_type = sensor_type
        self._name = entry.data.get(CONF_NAME, "Forecast")
        self._target_entity = entry.data.get(CONF_TARGET_ENTITY, "unknown")
        
        # Generate unique ID
        safe_name = self._name.lower().replace(" ", "_")
        self._attr_unique_id = f"nostradamus_{safe_name}_{sensor_type}"
        
        # Device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"Nostradamus: {self._name}",
            manufacturer="Nostradamus",
            model="Sensor Forecasting",
            sw_version="0.1.0",
        )

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.coordinator.data.get("status") == "ok"
        )


class NostradamusForecastSensor(NostradamusBaseSensor):
    """Sensor showing the next predicted value."""

    def __init__(
        self,
        coordinator: NostradamusCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the forecast sensor."""
        super().__init__(coordinator, entry, "forecast")
        
        self._attr_name = f"Nostradamus {self._name}"
        self._attr_icon = "mdi:crystal-ball"
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> Optional[float]:
        """Return the next predicted value."""
        if self.coordinator.data:
            return self.coordinator.data.get("next_value")
        return None

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement from the target entity."""
        # Try to get unit from target entity's current state
        if self._target_entity:
            state = self.hass.states.get(self._target_entity)
            if state:
                return state.attributes.get("unit_of_measurement")
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional state attributes."""
        attrs = {}
        
        if self.coordinator.data:
            data = self.coordinator.data
            
            attrs[ATTR_FORECAST] = data.get("forecast", [])
            attrs[ATTR_TIMESTAMPS] = data.get("timestamps", [])
            attrs[ATTR_LOWER_BOUND] = data.get("lower_bound", [])
            attrs[ATTR_UPPER_BOUND] = data.get("upper_bound", [])
            attrs[ATTR_CONFIDENCE_SCORE] = data.get("confidence_score")
            attrs[ATTR_LAST_TRAINED] = data.get("last_trained")
            attrs[ATTR_CADENCE] = data.get("cadence_seconds")
            attrs[ATTR_TARGET_ENTITY] = data.get("target_entity")
            attrs[ATTR_SUPPORTING_ENTITIES] = data.get("supporting_entities", [])
        
        return attrs


class NostradamusConfidenceSensor(NostradamusBaseSensor):
    """Sensor showing the forecast confidence score."""

    def __init__(
        self,
        coordinator: NostradamusCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the confidence sensor."""
        super().__init__(coordinator, entry, "confidence")
        
        self._attr_name = f"Nostradamus {self._name} Confidence"
        self._attr_icon = "mdi:gauge"
        self._attr_native_unit_of_measurement = PERCENTAGE
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> Optional[int]:
        """Return the confidence score."""
        if self.coordinator.data:
            return self.coordinator.data.get("confidence_score")
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional state attributes."""
        attrs = {}
        
        if self.coordinator.data:
            data = self.coordinator.data
            attrs[ATTR_LAST_TRAINED] = data.get("last_trained")
            attrs[ATTR_CADENCE] = data.get("cadence_seconds")
            attrs["training_samples"] = data.get("training_samples")
        
        return attrs

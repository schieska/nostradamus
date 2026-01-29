"""Config flow for Nostradamus integration."""

import logging
from typing import Any, Dict, Optional

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ADDON_HOST,
    CONF_HORIZON,
    CONF_SUPPORTING_ENTITIES,
    CONF_TARGET_ENTITY,
    DEFAULT_ADDON_HOST,
    DEFAULT_HORIZON,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class NostradamusConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Nostradamus."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: Dict[str, Any] = {}
        self._addon_connected: bool = False

    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step - check add-on connection."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            addon_host = user_input.get(CONF_ADDON_HOST, DEFAULT_ADDON_HOST)
            
            # Test connection to add-on
            connected = await self._test_addon_connection(addon_host)
            
            if connected:
                self._data[CONF_ADDON_HOST] = addon_host
                return await self.async_step_target()
            else:
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_ADDON_HOST, default=DEFAULT_ADDON_HOST): str,
            }),
            errors=errors,
            description_placeholders={
                "addon_url": DEFAULT_ADDON_HOST,
            },
        )

    async def async_step_target(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle target entity selection."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            target_entity = user_input.get(CONF_TARGET_ENTITY)
            name = user_input.get(CONF_NAME, "").strip()
            
            if not name:
                # Generate name from entity ID
                name = target_entity.split(".")[-1].replace("_", " ").title()
            
            self._data[CONF_TARGET_ENTITY] = target_entity
            self._data[CONF_NAME] = name
            
            return await self.async_step_supporting()

        return self.async_show_form(
            step_id="target",
            data_schema=vol.Schema({
                vol.Required(CONF_TARGET_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=SENSOR_DOMAIN,
                        multiple=False,
                    )
                ),
                vol.Optional(CONF_NAME, default=""): str,
            }),
            errors=errors,
        )

    async def async_step_supporting(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle supporting entities selection."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            supporting = user_input.get(CONF_SUPPORTING_ENTITIES, [])
            self._data[CONF_SUPPORTING_ENTITIES] = supporting
            
            return await self.async_step_horizon()

        return self.async_show_form(
            step_id="supporting",
            data_schema=vol.Schema({
                vol.Optional(CONF_SUPPORTING_ENTITIES, default=[]): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=[SENSOR_DOMAIN, "binary_sensor"],
                        multiple=True,
                    )
                ),
            }),
            errors=errors,
        )

    async def async_step_horizon(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle horizon configuration."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            horizon = user_input.get(CONF_HORIZON, DEFAULT_HORIZON)
            
            if horizon < 1:
                errors["base"] = "invalid_horizon"
            elif horizon > 168:  # Max 1 week
                errors["base"] = "horizon_too_large"
            else:
                self._data[CONF_HORIZON] = horizon
                
                # Create the config entry
                return await self._create_entry()

        return self.async_show_form(
            step_id="horizon",
            data_schema=vol.Schema({
                vol.Required(CONF_HORIZON, default=DEFAULT_HORIZON): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=168,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            errors=errors,
            description_placeholders={
                "target_entity": self._data.get(CONF_TARGET_ENTITY, "sensor"),
            },
        )

    async def _create_entry(self) -> FlowResult:
        """Create the config entry."""
        # Generate unique ID from target entity
        unique_id = f"nostradamus_{self._data[CONF_TARGET_ENTITY]}"
        
        await self.async_set_unique_id(unique_id)
        self._abort_if_unique_id_configured()
        
        title = f"Nostradamus: {self._data[CONF_NAME]}"
        
        return self.async_create_entry(
            title=title,
            data=self._data,
        )

    async def _test_addon_connection(self, host: str) -> bool:
        """Test connection to the Nostradamus add-on."""
        import aiohttp
        
        try:
            url = f"{host.rstrip('/')}/api/health"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "ok"
        except Exception as e:
            _LOGGER.warning(f"Failed to connect to add-on at {host}: {e}")
        
        return False

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Get the options flow for this handler."""
        return NostradamusOptionsFlow(config_entry)


class NostradamusOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Nostradamus."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current_horizon = self.config_entry.data.get(CONF_HORIZON, DEFAULT_HORIZON)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_HORIZON, default=current_horizon): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=168,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
        )

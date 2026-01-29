"""Home Assistant data fetcher via Supervisor API."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# API base URLs
SUPERVISOR_API = "http://supervisor/core/api"
SUPERVISOR_API_FALLBACK = os.environ.get("HA_URL", "http://homeassistant.local:8123") + "/api"


class HADataFetcher:
    """Fetches sensor data from Home Assistant via the Supervisor API."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            token: Supervisor token (from SUPERVISOR_TOKEN env) or long-lived access token
        """
        self.token = token or os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HA_TOKEN")
        
        # Determine API base URL
        if os.environ.get("SUPERVISOR_TOKEN"):
            self.api_base = SUPERVISOR_API
        else:
            self.api_base = SUPERVISOR_API_FALLBACK
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
    
    def get_states(self) -> List[Dict[str, Any]]:
        """Get all current entity states."""
        response = self.session.get(f"{self.api_base}/states")
        response.raise_for_status()
        return response.json()
    
    def get_numeric_entities(self) -> List[Dict[str, Any]]:
        """Get list of numeric sensor entities."""
        states = self.get_states()
        numeric_entities = []
        
        for state in states:
            entity_id = state.get("entity_id", "")
            
            # Filter for sensors
            if not entity_id.startswith(("sensor.", "binary_sensor.")):
                continue
            
            # Check if state is numeric
            state_value = state.get("state")
            if state_value in ("unavailable", "unknown", None):
                continue
            
            # Try to parse as number
            try:
                float(state_value)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = entity_id.startswith("binary_sensor.")
            
            if is_numeric:
                attrs = state.get("attributes", {})
                numeric_entities.append({
                    "entity_id": entity_id,
                    "friendly_name": attrs.get("friendly_name", entity_id),
                    "unit_of_measurement": attrs.get("unit_of_measurement"),
                    "state": state_value,
                    "device_class": attrs.get("device_class")
                })
        
        return numeric_entities
    
    def get_history(
        self,
        entity_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days_back: int = 14
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for entities.
        
        Args:
            entity_ids: List of entity IDs to fetch
            start_time: Start of history period (default: days_back days ago)
            end_time: End of history period (default: now)
            days_back: Number of days to look back if start_time not specified
            
        Returns:
            Dictionary mapping entity_id to DataFrame with timestamp and state columns
        """
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=days_back)
        
        # Format timestamps for API
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        
        # Build URL with entity filter
        entity_filter = ",".join(entity_ids)
        url = f"{self.api_base}/history/period/{start_str}"
        params = {
            "filter_entity_id": entity_filter,
            "end_time": end_str,
            "minimal_response": "true",
            "significant_changes_only": "false"
        }
        
        logger.info(f"Fetching history for {len(entity_ids)} entities from {start_str} to {end_str}")
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        history_data = response.json()
        
        result = {}
        for entity_history in history_data:
            if not entity_history:
                continue
            
            entity_id = entity_history[0].get("entity_id")
            if not entity_id:
                continue
            
            # Convert to DataFrame
            records = []
            for entry in entity_history:
                state = entry.get("state")
                last_changed = entry.get("last_changed")
                
                if state in ("unavailable", "unknown", None):
                    continue
                
                # Parse state
                try:
                    if entity_id.startswith("binary_sensor."):
                        value = 1.0 if state in ("on", "true", "1") else 0.0
                    else:
                        value = float(state)
                except (ValueError, TypeError):
                    continue
                
                # Parse timestamp
                try:
                    timestamp = pd.to_datetime(last_changed)
                except Exception:
                    continue
                
                records.append({"timestamp": timestamp, "value": value})
            
            if records:
                df = pd.DataFrame(records)
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
                df = df.set_index("timestamp")
                result[entity_id] = df
                logger.debug(f"Loaded {len(df)} records for {entity_id}")
        
        return result
    
    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a single entity."""
        response = self.session.get(f"{self.api_base}/states/{entity_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

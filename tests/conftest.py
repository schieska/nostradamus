"""Pytest configuration and fixtures for Nostradamus testing."""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

# Add the addon app to path for imports
addon_app_path = Path(__file__).parent.parent / "nostradamus_addon" / "rootfs" / "app"
sys.path.insert(0, str(addon_app_path))

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def ha_url() -> str:
    """Get Home Assistant URL from environment."""
    url = os.environ.get("HA_URL")
    if not url:
        pytest.skip("HA_URL environment variable not set")
    return url


@pytest.fixture(scope="session")
def ha_token() -> str:
    """Get Home Assistant token from environment."""
    token = os.environ.get("HA_TOKEN")
    if not token:
        pytest.skip("HA_TOKEN environment variable not set")
    return token


@pytest.fixture(scope="session")
def data_fetcher(ha_token: str):
    """Create a data fetcher connected to HA."""
    from nostradamus.data_fetcher import HADataFetcher
    return HADataFetcher(token=ha_token)


@pytest.fixture
def temp_storage(tmp_path: Path):
    """Create a temporary storage instance."""
    from nostradamus.storage import ForecastStorage
    return ForecastStorage(str(tmp_path))


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import pandas as pd
    import numpy as np
    
    # Generate 14 days of hourly data
    dates = pd.date_range(start="2026-01-01", periods=24 * 14, freq="h")
    
    # Create realistic temperature-like data with daily pattern
    hours = dates.hour
    base = 20  # Base temperature
    daily_variation = 5 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at 2pm
    noise = np.random.normal(0, 0.5, len(dates))
    
    values = base + daily_variation + noise
    
    df = pd.DataFrame({"value": values}, index=dates)
    return df


@pytest.fixture
def sample_binary_dataframe():
    """Create a sample binary sensor DataFrame."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start="2026-01-01", periods=24 * 14, freq="h")
    
    # Create binary data that's more likely to be 1 during day hours
    hours = dates.hour
    prob_on = 0.3 + 0.4 * ((hours >= 8) & (hours <= 22))
    values = (np.random.random(len(dates)) < prob_on).astype(float)
    
    df = pd.DataFrame({"value": values}, index=dates)
    return df

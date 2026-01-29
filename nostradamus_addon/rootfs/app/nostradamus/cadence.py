"""Automatic cadence detection for sensor data."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Common cadences in seconds
COMMON_CADENCES = [
    60,        # 1 minute
    300,       # 5 minutes
    600,       # 10 minutes
    900,       # 15 minutes
    1800,      # 30 minutes
    3600,      # 1 hour
    7200,      # 2 hours
    14400,     # 4 hours
    21600,     # 6 hours
    43200,     # 12 hours
    86400,     # 1 day
]


def detect_cadence(df: pd.DataFrame, max_samples: int = 1000) -> Tuple[int, float]:
    """
    Detect the most likely cadence (sampling interval) from time series data.
    
    Args:
        df: DataFrame with DatetimeIndex
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Tuple of (cadence_seconds, confidence_score)
        confidence_score is 0-1 indicating how well the data matches the detected cadence
    """
    if len(df) < 3:
        logger.warning("Not enough data points to detect cadence, defaulting to 1 hour")
        return 3600, 0.0
    
    # Get time deltas between consecutive points
    timestamps = df.index.to_series()
    if len(timestamps) > max_samples:
        timestamps = timestamps.tail(max_samples)
    
    deltas = timestamps.diff().dropna()
    
    # Convert to seconds
    delta_seconds = deltas.dt.total_seconds().values
    
    # Remove outliers (more than 3x median)
    median_delta = np.median(delta_seconds)
    filtered_deltas = delta_seconds[delta_seconds < median_delta * 3]
    
    if len(filtered_deltas) < 3:
        filtered_deltas = delta_seconds
    
    # Calculate statistics
    mean_delta = np.mean(filtered_deltas)
    std_delta = np.std(filtered_deltas)
    
    # Find the closest common cadence
    best_cadence = None
    best_distance = float("inf")
    
    for cadence in COMMON_CADENCES:
        distance = abs(mean_delta - cadence)
        if distance < best_distance:
            best_distance = distance
            best_cadence = cadence
    
    # If mean delta is very different from all common cadences, use the mean
    if best_distance > mean_delta * 0.5:
        # Round to nearest minute or 5 minutes
        if mean_delta < 300:
            best_cadence = max(60, round(mean_delta / 60) * 60)
        else:
            best_cadence = max(300, round(mean_delta / 300) * 300)
    
    # Calculate confidence score
    # High confidence if std is low relative to mean
    if mean_delta > 0:
        cv = std_delta / mean_delta  # Coefficient of variation
        confidence = max(0.0, min(1.0, 1.0 - cv))
    else:
        confidence = 0.0
    
    logger.info(
        f"Detected cadence: {best_cadence}s ({format_duration(best_cadence)}), "
        f"confidence: {confidence:.2%}"
    )
    
    return best_cadence, confidence


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    else:
        days = seconds // 86400
        return f"{days}d"


def resample_to_cadence(
    df: pd.DataFrame,
    cadence_seconds: int,
    method: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series to a fixed cadence.
    
    Args:
        df: DataFrame with DatetimeIndex and a 'value' column
        cadence_seconds: Target cadence in seconds
        method: Resampling method ('mean', 'last', 'first', 'max', 'min')
        
    Returns:
        Resampled DataFrame with regular intervals
    """
    if len(df) == 0:
        return df
    
    # Create rule string for pandas resample
    if cadence_seconds < 60:
        rule = f"{cadence_seconds}s"
    elif cadence_seconds < 3600:
        rule = f"{cadence_seconds // 60}min"
    elif cadence_seconds < 86400:
        rule = f"{cadence_seconds // 3600}h"
    else:
        rule = f"{cadence_seconds // 86400}D"
    
    # Resample
    resampler = df.resample(rule)
    
    if method == "mean":
        resampled = resampler.mean()
    elif method == "last":
        resampled = resampler.last()
    elif method == "first":
        resampled = resampler.first()
    elif method == "max":
        resampled = resampler.max()
    elif method == "min":
        resampled = resampler.min()
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    # Forward fill small gaps (up to 3 missing points)
    resampled = resampled.ffill(limit=3)
    
    # Drop any remaining NaN rows
    resampled = resampled.dropna()
    
    logger.debug(
        f"Resampled from {len(df)} to {len(resampled)} points at {format_duration(cadence_seconds)}"
    )
    
    return resampled


def get_required_history_days(horizon: int, cadence_seconds: int) -> int:
    """
    Calculate how many days of history are needed for training.
    
    We want at least 10x the horizon for training, plus some buffer.
    
    Args:
        horizon: Number of steps to forecast
        cadence_seconds: Cadence in seconds
        
    Returns:
        Number of days of history to fetch
    """
    horizon_seconds = horizon * cadence_seconds
    min_history_seconds = horizon_seconds * 10
    
    # Add buffer and convert to days
    days = max(7, int(min_history_seconds / 86400) + 3)
    
    # Cap at 60 days for practical reasons
    return min(60, days)

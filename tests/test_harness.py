"""
Interactive test harness for Nostradamus.

This script connects to a real Home Assistant instance and allows you to:
1. Browse available sensors
2. Test forecast generation on real data
3. Visualize forecasts

Usage:
    python tests/test_harness.py --entity sensor.power_usage --horizon 24
    python tests/test_harness.py --list-entities
    python tests/test_harness.py --entity sensor.temperature --visualize
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add addon app to path
addon_app_path = Path(__file__).parent.parent / "nostradamus_addon" / "rootfs" / "app"
sys.path.insert(0, str(addon_app_path))

from dotenv import load_dotenv

load_dotenv()


def get_data_fetcher():
    """Create a data fetcher with environment credentials."""
    from nostradamus.data_fetcher import HADataFetcher
    
    token = os.environ.get("HA_TOKEN")
    if not token:
        print("Error: HA_TOKEN environment variable not set")
        print("Set it in .env file or export HA_TOKEN=your-token")
        sys.exit(1)
    
    return HADataFetcher(token=token)


def list_entities():
    """List all numeric entities from Home Assistant."""
    print("Connecting to Home Assistant...")
    fetcher = get_data_fetcher()
    
    try:
        entities = fetcher.get_numeric_entities()
    except Exception as e:
        print(f"Error connecting to Home Assistant: {e}")
        sys.exit(1)
    
    print(f"\nFound {len(entities)} numeric entities:\n")
    
    # Group by domain
    by_domain = {}
    for entity in entities:
        entity_id = entity["entity_id"]
        domain = entity_id.split(".")[0]
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(entity)
    
    for domain, domain_entities in sorted(by_domain.items()):
        print(f"\n{domain.upper()} ({len(domain_entities)} entities)")
        print("-" * 50)
        for entity in domain_entities[:20]:  # Limit to 20 per domain
            name = entity.get("friendly_name", entity["entity_id"])
            unit = entity.get("unit_of_measurement", "")
            state = entity.get("state", "?")
            print(f"  {entity['entity_id']}")
            print(f"    {name}: {state} {unit}")
        
        if len(domain_entities) > 20:
            print(f"    ... and {len(domain_entities) - 20} more")


def test_forecast(entity_id: str, horizon: int, supporting_entities: list, visualize: bool):
    """Test forecast generation for an entity."""
    from nostradamus.data_fetcher import HADataFetcher
    from nostradamus.cadence import detect_cadence, resample_to_cadence, get_required_history_days
    from nostradamus.engine import ForecastEngine
    from nostradamus.storage import ForecastStorage
    
    print(f"\n{'='*60}")
    print(f"NOSTRADAMUS FORECAST TEST")
    print(f"{'='*60}")
    print(f"Target entity: {entity_id}")
    print(f"Supporting entities: {supporting_entities or 'None'}")
    print(f"Horizon: {horizon} steps")
    print(f"{'='*60}\n")
    
    # Create temporary storage
    temp_dir = Path("./test_data")
    temp_dir.mkdir(exist_ok=True)
    storage = ForecastStorage(str(temp_dir))
    
    # Create engine
    token = os.environ.get("HA_TOKEN")
    engine = ForecastEngine(supervisor_token=token, storage=storage)
    
    print("Step 1: Fetching entity history...")
    all_entities = [entity_id] + (supporting_entities or [])
    history = engine.data_fetcher.get_history(all_entities, days_back=14)
    
    if entity_id not in history:
        print(f"Error: No history found for {entity_id}")
        sys.exit(1)
    
    target_df = history[entity_id]
    print(f"  Loaded {len(target_df)} data points")
    print(f"  Date range: {target_df.index.min()} to {target_df.index.max()}")
    
    print("\nStep 2: Detecting cadence...")
    cadence, confidence = detect_cadence(target_df)
    cadence_str = format_cadence(cadence)
    print(f"  Detected cadence: {cadence_str}")
    print(f"  Confidence: {confidence:.1%}")
    
    print("\nStep 3: Creating forecast...")
    try:
        result = engine.create_forecast(
            forecast_id="test_forecast",
            name="Test Forecast",
            target_entity=entity_id,
            supporting_entities=supporting_entities or [],
            horizon=horizon
        )
    except Exception as e:
        print(f"Error creating forecast: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("FORECAST RESULTS")
    print(f"{'='*60}")
    print(f"Status: {result.get('status')}")
    print(f"Confidence Score: {result.get('confidence_score')}%")
    print(f"Training Samples: {result.get('training_samples')}")
    print(f"Next Value: {result.get('next_value')}")
    
    print(f"\nForecast ({horizon} steps):")
    forecast = result.get("forecast", [])
    timestamps = result.get("timestamps", [])
    lower = result.get("lower_bound", [])
    upper = result.get("upper_bound", [])
    
    print("-" * 60)
    print(f"{'Timestamp':<25} {'Forecast':>10} {'Lower':>10} {'Upper':>10}")
    print("-" * 60)
    
    for i in range(min(10, len(forecast))):  # Show first 10
        ts = timestamps[i][:19] if timestamps else "?"
        print(f"{ts:<25} {forecast[i]:>10.2f} {lower[i]:>10.2f} {upper[i]:>10.2f}")
    
    if len(forecast) > 10:
        print(f"  ... and {len(forecast) - 10} more steps")
    
    if visualize:
        visualize_forecast(target_df, result, cadence)
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"Results saved to: {temp_dir / 'forecasts' / 'test_forecast.json'}")


def format_cadence(seconds: int) -> str:
    """Format cadence as human-readable string."""
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        return f"{seconds // 60} minutes"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''}"
    else:
        days = seconds // 86400
        return f"{days} day{'s' if days > 1 else ''}"


def visualize_forecast(history_df, result, cadence_seconds):
    """Visualize the forecast with matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
    except ImportError:
        print("\nVisualization requires matplotlib. Install with:")
        print("  pip install matplotlib")
        return
    
    print("\nGenerating visualization...")
    
    # Prepare data
    forecast = result.get("forecast", [])
    timestamps = pd.to_datetime(result.get("timestamps", []))
    lower = result.get("lower_bound", [])
    upper = result.get("upper_bound", [])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot historical data (last 7 days)
    hist_start = history_df.index.max() - pd.Timedelta(days=7)
    hist_plot = history_df[history_df.index >= hist_start]
    
    ax.plot(hist_plot.index, hist_plot["value"], 
            label="Historical", color="blue", linewidth=1.5)
    
    # Plot forecast
    ax.plot(timestamps, forecast, 
            label="Forecast", color="red", linewidth=2, linestyle="--")
    
    # Plot confidence interval
    ax.fill_between(timestamps, lower, upper, 
                    alpha=0.2, color="red", label="90% Confidence")
    
    # Add vertical line at forecast start
    ax.axvline(x=history_df.index.max(), color="gray", linestyle=":", alpha=0.7)
    ax.text(history_df.index.max(), ax.get_ylim()[1], " Forecast →", 
            fontsize=10, color="gray")
    
    # Format
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"Nostradamus Forecast\n{result.get('target_entity', 'Unknown')} "
                 f"(Confidence: {result.get('confidence_score', '?')}%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save and show
    output_path = Path("./test_data/forecast_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    
    # plt.show()  # Disabled for non-interactive environments


def main():
    parser = argparse.ArgumentParser(
        description="Nostradamus Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all entities:
    python tests/test_harness.py --list-entities

  Test forecast for a sensor:
    python tests/test_harness.py --entity sensor.power_usage --horizon 24

  Test with supporting entities and visualization:
    python tests/test_harness.py --entity sensor.temperature \\
        --supporting sensor.outdoor_temp binary_sensor.window \\
        --horizon 48 --visualize
        """
    )
    
    parser.add_argument("--list-entities", action="store_true",
                        help="List available numeric entities")
    parser.add_argument("--entity", "-e", type=str,
                        help="Entity ID to forecast")
    parser.add_argument("--supporting", "-s", nargs="*", default=[],
                        help="Supporting entity IDs")
    parser.add_argument("--horizon", "-n", type=int, default=24,
                        help="Forecast horizon (number of steps)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Show visualization plot")
    
    args = parser.parse_args()
    
    if args.list_entities:
        list_entities()
    elif args.entity:
        test_forecast(args.entity, args.horizon, args.supporting, args.visualize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

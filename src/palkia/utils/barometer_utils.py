import numpy as np
import pandas as pd


def process_barometer_data(
    baro_data: pd.DataFrame,
    window_size: int = 50,
    pressure_threshold: float = 0.5,
) -> pd.DataFrame:
    """Process barometer data to identify stable periods and estimate relative altitude changes.

    Args:
    ----
        baro_data (pd.DataFrame): Raw barometer data.
        window_size (int): Size of the rolling window for smoothing.
        pressure_threshold (float): Threshold for identifying stable pressure periods.

    Returns:
    -------
        pd.DataFrame: Processed barometer data with altitude estimates.

    """
    # Smooth pressure data
    baro_data["smooth_pressure"] = (
        baro_data["pressure"].rolling(window=window_size, center=True).mean()
    )

    # Identify stable periods
    baro_data["pressure_diff"] = baro_data["smooth_pressure"].diff().abs()
    baro_data["is_stable"] = baro_data["pressure_diff"] < pressure_threshold

    # Estimate relative altitude changes
    reference_pressure = baro_data["smooth_pressure"].iloc[0]
    baro_data["relative_altitude"] = 44330 * (
        1 - (baro_data["smooth_pressure"] / reference_pressure) ** (1 / 5.255)
    )

    return baro_data


def detect_floor_changes(
    baro_data: pd.DataFrame,
    altitude_threshold: float = 2.5,
) -> pd.DataFrame:
    """Detect potential floor changes based on barometer data.

    Args:
    ----
        baro_data (pd.DataFrame): Processed barometer data.
        altitude_threshold (float): Threshold for detecting a floor change.

    Returns:
    -------
        pd.DataFrame: Data frame with detected floor changes.

    """
    floor_changes = []
    current_floor = 0

    for i in range(1, len(baro_data)):
        altitude_change = (
            baro_data["relative_altitude"].iloc[i]
            - baro_data["relative_altitude"].iloc[i - 1]
        )
        if abs(altitude_change) > altitude_threshold:
            current_floor += np.sign(altitude_change)
            floor_changes.append(
                {
                    "timestamp": baro_data["timestamp"].iloc[i],
                    "floor": int(current_floor),
                    "altitude_change": altitude_change,
                },
            )

    return pd.DataFrame(floor_changes)

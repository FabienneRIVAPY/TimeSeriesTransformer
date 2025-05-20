import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    """
    Load data from CSV file and clean it by replacing zeros with NaN values.

    Parameters:
    file_path (str): Path to the CSV file containing data

    Returns:
    pandas.DataFrame: Cleaned DataFrame with datetime index and NaN values
    """
    # Read the CSV file with proper parameters
    data = pd.read_csv(
        file_path,
        index_col=0,  # Use first column as index
        sep=";",  # Use semicolon as separator
        decimal=",",  # Use comma as decimal point
    )

    # Convert index to datetime format
    data.index = pd.to_datetime(data.index, dayfirst=True)

    # Sort the index chronologically
    data.sort_index(inplace=True)

    # Replace all 0.0 values with NaN
    data.replace(0.0, np.nan, inplace=True)

    return data


def process_timeseries(
    df: pd.DataFrame,
    freq: "str" = "1h",
    time_window: tuple = (1096, 1346),
) -> pd.DataFrame:
    """
    Process time series data with missing value handling and feature engineering.

    Parameters:
    ----------
    data : pd.DataFrame
        Input DataFrame containing time series data

    Returns:
    -------
    pd.DataFrame
        Processed DataFrame with extracted features and cleaned data
    """
    # Resample to hourly frequency and calculate mean
    df = df.resample(freq).mean().replace(0.0, np.nan)

    # Get earliest time reference point
    earliest_time = df.index.min()

    df_list = []
    # Select specific column and handle missing values
    for label in df:
        data = df

        # Forward fill missing values
        data = data.ffill()

        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.ffill()

        # Extract time series and determine valid date range
        ts = data[label]
        start_date = min(ts.fillna(method="ffill").dropna().index)
        end_date = max(ts.fillna(method="bfill").dropna().index)

        # Create processed DataFrame
        tmp = pd.DataFrame({"DAprices": data[label]})
        date = tmp.index

        # Filter to valid date range
        active_range = (ts.index >= start_date) & (ts.index <= end_date)
        ts = ts[active_range].fillna(0.0)

        # Add temporal features
        tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time
        ).days * 24
        tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["date"] = date
        tmp["zone"] = label
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month

        # stack all time series vertically
        df_list.append(tmp)

    time_df = pd.concat(df_list).reset_index(drop=True)
    # Apply final filtering
    # time_df = tmp[
    #     (tmp["days_from_start"] >= time_window[0])
    #     & (tmp["days_from_start"] < time_window[1])
    # ].copy()

    return time_df

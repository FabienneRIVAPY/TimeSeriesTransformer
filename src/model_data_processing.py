import holidays
import pandas as pd
import numpy as np


def save_path_to_file(path_str, output_filename):
    """
    Write a path string directly to a specified output file.

    Args:
        path_str (str): The path string to be saved
        output_filename (str): The filename where the path will be written

    Returns:
        None
    """
    # Write to file directly
    with open(output_filename, "w") as f:
        f.write(path_str)


def mark_holidays_and_weekends(date):
    """
    Returns True if the date is either a German holiday or a weekend
    """
    # Create German holidays object
    german_holidays = holidays.Germany()

    # Check if it's a weekend (Saturday=5, Sunday=6)
    is_weekend = date.weekday() >= 5

    # Check if it's a holiday
    is_holiday = date in german_holidays

    if is_weekend or is_holiday:
        return 1
    else:
        return 0


def preprocess_input_data(data):
    data.index = pd.to_datetime(data.index, dayfirst=True)

    data.sort_index(inplace=True)
    data.replace(0.0, np.nan)
    df_list = []
    # data = data.resample("1h").mean().replace(0.0, np.nan)
    earliest_time = data.index.min()
    # data = data[["Day Ahead Preise D_LU"]]
    # Forward fill missing values
    data = data.ffill()
    # Handle remaining infinite values
    data = data.replace("-", np.nan)
    data = data.replace("#VALUE!", np.nan)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()

    ts = data[["Day Ahead Preise D_LU"]]

    start_date = min(ts.fillna(method="ffill").dropna().index)
    end_date = max(ts.fillna(method="bfill").dropna().index)

    tmp = data  # pd.DataFrame(
    # {"DAprices": data["Deutschland/Luxemburg [€/MWh] Originalauflösungen"]}
    # )
    date = date = tmp.index

    active_range = (ts.index >= start_date) & (ts.index <= end_date)
    ts = ts[active_range].fillna(0.0)

    tmp["hours_from_start"] = (date - earliest_time).seconds / 60 / 60 + (
        date - earliest_time
    ).days * 24
    tmp["hours_from_start"] = tmp["hours_from_start"].astype("int")

    tmp["days_from_start"] = (date - earliest_time).days

    #     #tmp['days_from_start'] = (date - earliest_time).days
    tmp["date"] = date
    tmp["zone"] = "Deutschland/Luxemburg [€/MWh] Originalauflösungen"
    tmp["hour"] = date.hour
    tmp["day"] = date.day
    tmp["day_of_week"] = date.dayofweek
    tmp["month"] = date.month
    tmp["year"] = date.year
    tmp["is_holiday_or_weekend"] = tmp["date"].apply(mark_holidays_and_weekends)
    tmp["GWL"] = 0.0

    df_list.append(tmp)

    time_df = pd.concat(df_list).reset_index(drop=True)
    return time_df

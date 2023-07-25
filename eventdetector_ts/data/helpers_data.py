import json
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd
from dateutil.parser import parser
# noinspection PyUnresolvedReferences
from numpy.lib.stride_tricks import as_strided
from pandas.core.dtypes.common import is_datetime64_any_dtype

from eventdetector_ts import TIME_LABEL, FILL_NAN_ZEROS, FILL_NAN_FFILL, FILL_NAN_BFILL, FILL_NAN_MEDIAN, \
    MIDDLE_EVENT_LABEL, TimeUnit
from eventdetector_ts.data.interval import Interval


def overlapping_partitions(data: np.ndarray, width: int, step: int = 1):
    """
    Splits an input numpy array into a set of overlapping partitions.

    Args:
        data: Input numpy array to be split into overlapping partitions
        width: Width of each overlapping partition
        step: The step size between successive partitions (default=1)

    Returns:
        Numpy array of shape (nb_partitions, width, data.ndim), containing the created overlapping partitions.
    """
    if width > data.shape[0]:
        raise ValueError("Partition size cannot be greater than the size of the input data")
    if step > width:
        raise ValueError("Step size cannot be greater than partition size")

    # Compute the parameters for creating the overlapping partitions
    np_partitions = (data.shape[0] - width) // step + 1
    shape = (np_partitions, width) + data.shape[1:]
    strides = (step * data.strides[0],) + data.strides

    # Use as_strided to create the overlapping partitions
    partitioned_array = as_strided(data, shape=shape, strides=strides)

    return partitioned_array


def convert_dataframe_to_overlapping_partitions(
        dataframe: pd.DataFrame,
        width: int,
        step: int,
        fill_method: Optional[str] = None
) -> np.ndarray:
    """
    Converts a given DataFrame to overlapping partitions.

    Args:
        dataframe: Input DataFrame of features
        width: Width of each overlapping partition
        step: The step size between successive partitions
        fill_method: The method to use for filling NaNs. Supported methods are 'zeros', 'ffill', 'bfill', and 'median'.
            If None, NaNs are left as-is. (default=None)

    Returns:
        Numpy array of shape (np_partitions, width, nb_features), containing the created overlapping partitions.
    """

    dataframe = dataframe.copy()
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.loc[:, TIME_LABEL] = dataframe.index.to_pydatetime()

    if fill_method == FILL_NAN_ZEROS:
        dataframe = dataframe.fillna(0)
    elif fill_method == FILL_NAN_FFILL:
        dataframe = dataframe.ffill()
    elif fill_method == FILL_NAN_BFILL:
        dataframe = dataframe.bfill()
    elif fill_method == FILL_NAN_MEDIAN:
        dataframe = dataframe.fillna(dataframe.median())
    elif fill_method is not None:
        raise ValueError(f"Unsupported fill method: {fill_method}")

    sw = overlapping_partitions(dataframe.to_numpy(), width=width, step=step)
    return sw


class InvalidArgumentError(ValueError):
    """Raised when an invalid argument is passed to a function or method."""

    def __init__(self, message):
        """
        Initialize a new InvalidArgumentError with the specified error message.

        Args:
            message (str): The error message to display.
        """
        super().__init__(message)


def convert_time_to_datetime(date: Union[str, pd.Timestamp, float, int], to_timestamp: bool = True) -> \
        Union[float, datetime]:
    """
    Converts a date string, pandas Timestamp, or numeric timestamp to a Python datetime or Unix timestamp.

    Args:
        date: The input date as a string, pandas Timestamp, or numeric timestamp.
        to_timestamp: If True (default), return the date as a Unix timestamp (float), otherwise as a Python datetime.

    Returns:
        The input date as a Unix timestamp or Python datetime object.
    """

    if isinstance(date, pd.Timestamp):
        dt = date.to_pydatetime()
    elif isinstance(date, (float, int)):
        dt = datetime.fromtimestamp(date)
    elif isinstance(date, str):
        dt = parser.parse(date, ignoretz=True)
    else:
        raise ValueError(f"Invalid date format {date}. Supported formats are str, pd.Timestamp, float, and int.")

    if to_timestamp:
        return (dt - datetime(1970, 1, 1)).total_seconds()
    return dt


def num_columns(lst: list) -> int:
    """
    Returns the number of columns in a list.

    Args:
        lst (list): The list to check.

    Returns:
        int: The number of columns in the list.
    """

    if not lst:
        # if the list is empty return 0
        return 0
    elif isinstance(lst[0], list):
        # if the first element of the list is a list, return the length of the first list
        return len(lst[0])
    else:
        # otherwise return 1, because the list has only one column
        return 1


def compute_middle_event(events: Union[list, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes the middle date of events and returns it as a DataFrame.

    Args:
        events (Union[list, pd.DataFrame]): A list or pandas DataFrame containing the starting and ending
            dates of events.

    Returns:
        pd.DataFrame: A pandas DataFrame with a single column containing the middle dates of events.
    """
    column1 = "Starting Date"
    column2 = "Ending Date"
    is2d = True

    if isinstance(events, list):
        nb_columns = num_columns(events)
        if nb_columns == 2:
            df = pd.DataFrame(events, columns=[column1, column2])
        elif nb_columns == 1:
            df = pd.DataFrame(events, columns=[column1])

            is2d = False
        else:
            raise ValueError(
                f"The list of events is not compatible. The number of columns {nb_columns} should not exceed 2.")
    elif isinstance(events, pd.DataFrame):
        df = events
        columns = events.columns
        if len(columns) == 2:
            df = df.rename(columns={columns[0]: column1, columns[1]: column2})
        elif len(columns) == 1:
            is2d = False
            df = df.rename(columns={columns[0]: column1})
        else:
            raise ValueError("The dataframe of events in not compatible, columns should not exceed 2")
    else:
        raise ValueError("The events argument must be a list or pandas DataFrame.")

    df[column1] = pd.to_datetime(df[column1])
    if is2d:
        df[column2] = pd.to_datetime(df[column2])

    if is2d:
        df[column1] = df[column1].apply(lambda x: convert_time_to_datetime(x) / 2)
        df[column2] = df[column2].apply(lambda x: convert_time_to_datetime(x) / 2)
        df[MIDDLE_EVENT_LABEL] = df[column1] + df[column2]
    else:
        df[MIDDLE_EVENT_LABEL] = df[column1].apply(lambda x: convert_time_to_datetime(x))

    df[MIDDLE_EVENT_LABEL] = df[MIDDLE_EVENT_LABEL].apply(lambda x: datetime.utcfromtimestamp(x))
    df = df[[MIDDLE_EVENT_LABEL]]
    df = df.sort_values(by=MIDDLE_EVENT_LABEL)
    return df


def remove_close_events(events_df: pd.DataFrame, delta_unit_time: int, unit: TimeUnit) -> pd.DataFrame:
    """
    Removes events from a DataFrame that occur too close together.

    Args:
        unit: The time unit
        events_df: A pandas DataFrame containing events with a column named 'middle_event'.
        delta_unit_time: A integer representing the minimum time in unit time between events.

    Returns:
        A pandas DataFrame with close events removed.
    """

    # Convert delta to timedelta
    delta = get_timedelta(delta_unit_time, unit)

    # List to hold indices of events to delete
    events_to_delete = []

    # Loop through all events
    for i in range(len(events_df)):
        # Get middle time of the current event
        mid_time = events_df.iloc[i][MIDDLE_EVENT_LABEL]

        # Skip current event if it's already marked for deletion
        if i in events_to_delete:
            continue

        # Loop through all remaining events
        for j in range(i + 1, len(events_df)):
            # Get middle time of the next event
            mid_time1 = events_df.iloc[j][MIDDLE_EVENT_LABEL]

            # If the next event is too close to the current event, mark it for deletion
            if (mid_time1 - mid_time) <= delta:
                events_to_delete.append(j)
            else:
                break

    # Drop events that were marked for deletion
    return events_df.drop(events_df.index[events_to_delete])


def convert_events_to_intervals(events_df: pd.DataFrame, w_s: int, unit: TimeUnit) -> list[Interval]:
    """
    Convert events from a pandas DataFrame to intervals.

    Args:
        events_df (pd.DataFrame): DataFrame containing the events' data.
        w_s (int): The overlapping partition size in unit time.
        unit: The unit time

    Returns:
        list[Interval]: A list of intervals.

    """
    # Create an empty list to store the intervals
    events_intervals = []

    # Loop over the events in the DataFrame
    for i in range(len(events_df)):
        # Get the middle event time
        mid_time = events_df.iloc[i][MIDDLE_EVENT_LABEL]

        # Compute the radius of the interval based on the overlapping partition size
        radius = get_timedelta(delta_unit_time=w_s // 2, unit=unit)

        # Create an interval with the middle event time at the center
        interval = Interval(mid_time - radius, mid_time + radius)

        # Add the interval to the list of intervals
        events_intervals.append(interval)

    # Return the list of intervals
    return events_intervals


def get_union_times_events(events_df: pd.DataFrame, partition_size: int, unit_time: TimeUnit) -> pd.DatetimeIndex:
    """
    Given a DataFrame of events and a time partition size in unit time, computes a DatetimeIndex of all times during 
    which at least one event was taking place.

    Args:
        events_df (pd.DataFrame): A DataFrame containing at least a MIDDLE_EVENT_LABEL column with the datetime
            of each event.
        partition_size (int): The size of the time partition to consider before and after each event.
        unit_time (TimeUnit): The unit time

    Returns:
        pd.DatetimeIndex: A DatetimeIndex of all times during which at least one event was taking place.
    """

    times_during_events = []
    previous_range = None
    for i, event_time in enumerate(events_df[MIDDLE_EVENT_LABEL]):
        start_time = event_time - get_timedelta(partition_size, unit=unit_time)
        end_time = event_time + get_timedelta(partition_size, unit=unit_time)
        # Generate a list of dates between start_time and end_time with a frequency of exactly (end_time - start_time).
        # This ensures that the last date is exactly equal to end_time (useful when we generate overlapping ranges).
        dates_between = pd.date_range(start=start_time, end=end_time, freq=end_time - start_time)

        if previous_range is None:
            times_during_events.append(dates_between)
            previous_range = dates_between
        else:
            # Check if the current range overlaps with the previous one.
            ranges_overlap = max(previous_range[0], previous_range[-1]) < min(dates_between[0], dates_between[-1])
            if not ranges_overlap:
                # If the ranges don't overlap, then we need to merge the previous and current ranges.
                merged_range = pd.date_range(start=previous_range[0], end=dates_between[-1],
                                             freq=dates_between[-1] - previous_range[0])
                # Replace the last range we added to the list with the merged range.
                times_during_events[-1] = merged_range
                previous_range = merged_range
            else:
                previous_range = dates_between
                times_during_events.append(dates_between)

    # Use the reduce function to combine all the overlapping ranges we generated.
    union_ranges = reduce(lambda x, y: x.union(y), times_during_events)
    # Remove any timezone information from the resulting DatetimeIndex, if present.
    union_ranges = union_ranges.tz_localize(None)
    return union_ranges


def get_dataset_within_events_times(data_set: pd.DataFrame, events_times: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extracts the data from the given dataset that falls within the specified event times.

    Args:
        data_set: A pandas DataFrame containing the data to extract.
        events_times: A pandas DatetimeIndex containing the times of events.

    Returns:
        A pandas DataFrame containing the data within the specified event times.
    """

    dataset_within_events_times = []

    # Iterate through the event times by pairs
    for i in range(0, len(events_times) - 1, 2):
        partition_start_time = events_times[i]
        partition_end_time = events_times[i + 1]

        # Extract the data within the event time
        data_within_event_time = data_set.loc[partition_start_time: partition_end_time]

        dataset_within_events_times.append(data_within_event_time)

    # Concatenate all the data extracted from events times
    return pd.concat(dataset_within_events_times)


def op(dataset_as_overlapping_partitions: np.ndarray, events_as_intervals: list[Interval]) -> \
        tuple[np.ndarray, np.ndarray]:
    """
    Calculates the "op" value for each overlapping partition in the dataset, based on the overlapping parameter 
    between the partition and a set of events.

    Args: dataset_as_overlapping_partitions: A numpy ndarray containing the overlapping partitions for the dataset, 
    where each overlapping partition is a 2D numpy ndarray containing the data points for the partition and their 
    timestamps. events_as_intervals: A list of Interval objects representing the events in the dataset.

    Returns:
        A tuple containing two values:
            - A numpy ndarray containing the overlapping partitions for the dataset, with the timestamp column removed.
            - A numpy ndarray of floating-point values representing the "op" value
                for each overlapping partition in the dataset.
    """

    # The index of the first event that hasn't been checked yet
    starting_event_index = 0

    # List to store the calculated op values for each overlapping partition
    op_values = []

    # Iterate through each overlapping partition in the dataset
    for partition in dataset_as_overlapping_partitions:
        # Get the start and end times of the current overlapping partition
        partition_start_time = partition[0][-1].to_pydatetime()
        partition_end_time = partition[-1][-1].to_pydatetime()

        # Create an Interval object to represent the current overlapping partition
        partition_interval = Interval(partition_start_time, partition_end_time)

        # Initialize the op value for the current overlapping partition to 0
        current_op_value = 0

        # Iterate through each event that hasn't been checked yet
        for event_index in range(starting_event_index, len(events_as_intervals)):
            # Get the Interval object for the current event
            current_event_interval = events_as_intervals[event_index]

            # If the start time of the current partition is greater than or equal to the end time of the current event,
            # we can skip this event since it doesn't overlap with the current partition
            if partition_interval.start_time >= current_event_interval.end_time:
                starting_event_index = event_index + 1
                continue

            # Calculate the overlapping parameter between the current partition and the current event
            overlapping_parameter = partition_interval.overlapping_parameter(current_event_interval)

            # If the overlapping parameter is 0, there is no overlap between the current partition and the current event
            if overlapping_parameter == 0:
                break

            # Update the op value for the current partition if the overlapping parameter is greater than the current op
            # value
            if overlapping_parameter > current_op_value:
                current_op_value = overlapping_parameter

        # Add the op value for the current partition to the list of op values
        op_values.append(current_op_value)

    # Remove the column containing the timestamps from the overlapping partitions
    dataset_as_overlapping_partitions = np.delete(dataset_as_overlapping_partitions, -1, axis=2)

    # Return the updated overlapping partitions and the op values
    return dataset_as_overlapping_partitions, np.array(op_values)


def get_timedelta(delta_unit_time: int, unit: TimeUnit) -> timedelta:
    """
    Returns a timedelta object with the specified delta_unit_time in the specified TimeUnit.

    Args:
        delta_unit_time: The delta unit time value.
        unit: The TimeUnit enum value representing the unit of time.

    Returns:
        A timedelta object with the specified delta_unit_time in the specified TimeUnit.
    """
    if unit == TimeUnit.MICROSECOND:
        return timedelta(microseconds=delta_unit_time)
    elif unit == TimeUnit.MILLISECOND:
        return timedelta(milliseconds=delta_unit_time)
    elif unit == TimeUnit.SECOND:
        return timedelta(seconds=delta_unit_time)
    elif unit == TimeUnit.MINUTE:
        return timedelta(minutes=delta_unit_time)
    elif unit == TimeUnit.HOUR:
        return timedelta(hours=delta_unit_time)
    elif unit == TimeUnit.DAY:
        return timedelta(days=delta_unit_time)
    elif unit == TimeUnit.YEAR:
        return timedelta(days=delta_unit_time * 365)
    else:
        raise ValueError("Invalid TimeUnit value.")


def get_total_units(timedelta_: timedelta, unit: TimeUnit) -> float:
    if unit == TimeUnit.MICROSECOND:
        return timedelta_.total_seconds() * 1e6
    elif unit == TimeUnit.MILLISECOND:
        return timedelta_.total_seconds() * 1e3
    elif unit == TimeUnit.SECOND:
        return timedelta_.total_seconds()
    elif unit == TimeUnit.MINUTE:
        return timedelta_.total_seconds() / 60
    elif unit == TimeUnit.HOUR:
        return timedelta_.total_seconds() / 3600
    elif unit == TimeUnit.DAY:
        return timedelta_.total_seconds() / (3600 * 24)
    elif unit == TimeUnit.YEAR:
        return timedelta_.total_seconds() / (3600 * 24 * 365.25)
    else:
        raise ValueError("Invalid TimeUnit value.")


def check_time_unit(diff: timedelta) -> Tuple[int, TimeUnit]:
    """
    Method to determine the unit of time of the dataset.

    Args:
        diff (timedelta): The time difference to be checked.

    Returns:
        Tuple[int, TimeUnit]: A tuple with the time value and its unit.
    """

    if diff.total_seconds() >= 31536000:  # 1 year in seconds
        years = int(diff.total_seconds() / 31536000)
        t_s = years
        time_unit = TimeUnit.YEAR
    elif diff.total_seconds() >= 86400:  # 1 day in seconds
        days = int(diff.total_seconds() / 86400)
        t_s = days
        time_unit = TimeUnit.DAY
    elif diff.total_seconds() >= 3600:  # 1 hour in seconds
        hours = int(diff.total_seconds() / 3600)
        t_s = hours
        time_unit = TimeUnit.HOUR
    elif diff.total_seconds() >= 60:  # 1 minute in seconds
        minutes = int(diff.total_seconds() / 60)
        t_s = minutes
        time_unit = TimeUnit.MINUTE
    elif diff.total_seconds() >= 1:
        t_s = int(diff.total_seconds())
        time_unit = TimeUnit.SECOND
    elif diff.total_seconds() * 1000 >= 1:
        t_s = int(diff.total_seconds() * 1000)
        time_unit = TimeUnit.MILLISECOND
    elif diff.total_seconds() * 1000000 >= 1:
        t_s = int(diff.total_seconds() * 1000000)
        time_unit = TimeUnit.MICROSECOND
    else:
        raise ValueError("Could not determine the unit of time of the dataset")

    return t_s, time_unit


def save_dict_to_json(path: str, data: Dict):
    """
    Save a dictionary into a json file
    Args:
        path (str): the path where to store the json file
        data (Dict): the dictionary

    Returns:

    """
    with open(path, 'w') as f:
        json.dump(data, f)


def convert_dataset_index_to_datetime(dataset: pd.DataFrame) -> None:
    """
    Check if the index of the DataFrame dataset is already in the datetime format. If the index is not in datetime 
    format, dataset.index = pd.to_datetime(dataset.index) statement is executed to convert it. 
    
    Args: 
        dataset (pd.DataFrame): A dataset as pandas DataFrame

    Returns:
        None
    """
    if not is_datetime64_any_dtype(dataset.index):
        dataset.index = pd.to_datetime(dataset.index)

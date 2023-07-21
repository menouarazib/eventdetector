from datetime import datetime

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from eventdetector_ts import TimeUnit
from eventdetector_ts.data.helpers import convert_time_to_datetime, get_timedelta


def event_to_rectangle(event, w_s, time_unit: TimeUnit, color, height=1, style="solid"):
    """
    Function to convert an event to a rectangle object for visualization.
    
    Args:
    event (datetime or other): The event timestamp or object.
    w_s (int): The partition size in time units.
    time_unit (TimeUnit): The time unit of the partition size.
    color (str): The color of the rectangle.
    height (int): The height of the rectangle.
    style (str): The linestyle of the rectangle.

    Returns:
        Rectangle: The rectangle object representing the event.

    """
    time = event
    if not isinstance(event, datetime):
        time = convert_time_to_datetime(event, to_timestamp=False)
    w_s_timedelta = get_timedelta(w_s // 2, time_unit)
    start_time = time - w_s_timedelta
    end_time = time + w_s_timedelta

    start_rect = mdates.date2num(start_time)
    end_rect = mdates.date2num(end_time)

    width_rect = end_rect - start_rect
    rect = Rectangle((start_rect, 0), width_rect, height, edgecolor=color, linestyle=style,
                     facecolor='none', linewidth=1)

    return rect

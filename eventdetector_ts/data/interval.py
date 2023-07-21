from datetime import datetime, timedelta


class Interval:
    """
    Represents a time interval between two datetime objects. This class is used to model an event or partition in
    time-series.
    """

    def __init__(self, start_time: datetime, end_time: datetime):
        """
        Constructs an interval for a given start and end time.

        Args:
            start_time (datetime): The starting time of the interval.
            end_time (datetime): The ending time of the interval.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time

    def __str__(self) -> str:
        """
        Returns a string representation of the interval in the format "start_time ---> end_time".

        Returns:
            str: A string representation of the interval.
        """
        return "{} ---> {}".format(self.start_time, self.end_time)

    def overlap(self, other: 'Interval') -> timedelta:
        """
        Computes the overlapping time (ot) between this interval and another interval.

        Args:
            other (Interval): Another interval to compare with.

        Returns:
            timedelta: The overlapping time between this interval and the other interval as a timedelta object.
        """
        overlap_start_time = max(self.start_time, other.start_time)
        overlap_end_time = min(self.end_time, other.end_time)
        overlap_duration = max(timedelta(0), overlap_end_time - overlap_start_time)
        return overlap_duration

    def overlapping_parameter(self, other: 'Interval') -> float:
        """
        Computes the overlapping parameter between this interval and another interval.

        Args:
            other (Interval): Another interval to compare with.

        Returns:
            float: A floating number between 0.0 and 1.0 representing the degree of overlap between the two intervals.
        """
        if other is None:
            return 0.0
        overlap_duration = self.overlap(other)
        total_duration = self.duration + other.duration - overlap_duration
        return overlap_duration / total_duration

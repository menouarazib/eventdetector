import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sympy.testing import pytest

from eventdetector_ts import TimeUnit
from eventdetector_ts.data.helpers import overlapping_partitions, compute_middle_event, \
    num_columns, convert_dataframe_to_overlapping_partitions, get_timedelta


def test_overlapping_partitions():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected_output = np.array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]], [[7, 8, 9], [10, 11, 12]]])
    assert np.array_equal(overlapping_partitions(data, width=2, step=1), expected_output)


class TestHelpers(unittest.TestCase):

    def setUp(self):
        self.n: int = 100

    def test_overlapping_partitions(self):
        # Test case 1: 1D input
        data1 = np.array([1, 2, 3, 4, 5])
        result1 = overlapping_partitions(data1, width=3, step=1)
        expected1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.assertTrue(np.array_equal(result1, expected1))

        # Test case 2: partition width greater than the size of the input data
        data2 = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            overlapping_partitions(data2, width=6, step=1)

        # Test case 3: 2D input
        data3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result3 = overlapping_partitions(data3, width=2, step=1)
        expected3 = np.array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]]])
        assert np.array_equal(result3, expected3)

        # Test case 4: 2D input
        data4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result4 = overlapping_partitions(data4, width=2, step=2)
        expected4 = np.array([[[1, 2, 3], [4, 5, 6]]])
        assert np.array_equal(result4, expected4)

    def test_convert_dataframe_to_overlapping_partitions(self):
        # Create a sample DataFrame with datetime index and real-valued features

        data = np.random.rand(self.n, 3)
        index = pd.date_range(start='2022-01-01', periods=self.n, freq='D')
        df = pd.DataFrame(data=data, columns=['feat1', 'feat2', 'feat3'], index=index)

        # Test overlapping partition generation with default settings
        sw = convert_dataframe_to_overlapping_partitions(df, width=2, step=1)
        expected_shape = (self.n - 1, 2, 4)  # Number of partitions, partition width, number of features+time
        self.assertEqual(sw.shape, expected_shape)

        # Test overlapping partition generation with custom settings
        sw = convert_dataframe_to_overlapping_partitions(df, width=14, step=7, fill_method='ffill')
        expected_shape = (13, 14, 4)  # Number of partitions, partition width, number of features+time
        self.assertEqual(sw.shape, expected_shape)

    def test_compute_middle_event(self):
        # Test case 1: List of events with 2 columns
        events_list = [['2022-01-01', '2022-01-02'], ['2022-01-03', '2022-01-05']]
        expected_output = pd.DataFrame({'event': [datetime(2022, 1, 1, 12, 0), datetime(2022, 1, 4)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_list)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)

        # Test case 2: List of events with 1 column
        events_list = [['2022-01-01'], ['2022-01-03']]
        expected_output = pd.DataFrame({"event": [datetime(2022, 1, 1), datetime(2022, 1, 3)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_list)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)
        # Test case 3: Pandas DataFrame with 2 columns
        events_df = pd.DataFrame({'Starting Date': ['2022-01-01', '2022-01-03'],
                                  'Ending Date': ['2022-01-02', '2022-01-05']})
        expected_output = pd.DataFrame({"event": [datetime(2022, 1, 1, 12, 0), datetime(2022, 1, 4)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_df)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)

        # Test case 4: Pandas DataFrame with 1 column
        expected_output = pd.DataFrame({"event": [datetime(2022, 1, 1), datetime(2022, 1, 3)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_list)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)

        # Test case 5: Empty list of events
        events_list = []
        with pytest.raises(ValueError):
            compute_middle_event(events_list)

        # Test case 6: Empty DataFrame of events
        events_df = pd.DataFrame()
        with pytest.raises(ValueError):
            compute_middle_event(events_df)

        # Test case 7: Invalid input format for events
        events_list = [[1, 2], [3, 4, 5]]
        with pytest.raises(ValueError):
            compute_middle_event(events_list)

    def test_empty_list(self):
        self.assertEqual(num_columns([]), 0)

    def test_single_column_list(self):
        self.assertEqual(num_columns([1, 2, 3]), 1)

    def test_multi_column_list(self):
        self.assertEqual(num_columns([[1, 2], [3, 4], [5, 6]]), 2)

    def test_mixed_list(self):
        self.assertEqual(num_columns([[1, 2], 3, 4]), 2)

    def test_microsecond(self):
        result = get_timedelta(100, TimeUnit.MICROSECOND)
        self.assertEqual(result, timedelta(microseconds=100))

    def test_millisecond(self):
        result = get_timedelta(500, TimeUnit.MILLISECOND)
        self.assertEqual(result, timedelta(milliseconds=500))

    def test_second(self):
        result = get_timedelta(60, TimeUnit.SECOND)
        self.assertEqual(result, timedelta(seconds=60))

    def test_minute(self):
        result = get_timedelta(30, TimeUnit.MINUTE)
        self.assertEqual(result, timedelta(minutes=30))

    def test_hour(self):
        result = get_timedelta(2, TimeUnit.HOUR)
        self.assertEqual(result, timedelta(hours=2))

    def test_day(self):
        result = get_timedelta(5, TimeUnit.DAY)
        self.assertEqual(result, timedelta(days=5))

    def test_year(self):
        result = get_timedelta(2, TimeUnit.YEAR)
        self.assertEqual(result, timedelta(days=2 * 365))

    def test_invalid_unit(self):
        with self.assertRaises(ValueError):
            get_timedelta(10, "invalid_unit")


if __name__ == '__main__':
    unittest.main()

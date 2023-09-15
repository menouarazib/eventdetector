import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype
from sympy.testing import pytest

from eventdetector_ts import TimeUnit
from eventdetector_ts.data.helpers_data import overlapping_partitions, compute_middle_event, \
    num_columns, convert_dataframe_to_overlapping_partitions, get_timedelta, get_total_units, check_time_unit, \
    convert_dataset_index_to_datetime, convert_seconds_to_time_unit


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
            get_timedelta(10, "null")

    def test_microsecond_(self):
        td = timedelta(microseconds=123456789)
        self.assertEqual(get_total_units(td, TimeUnit.MICROSECOND), 123456789)

    def test_millisecond_(self):
        td = timedelta(milliseconds=123456)
        self.assertEqual(get_total_units(td, TimeUnit.MILLISECOND), 123456)

    def test_second_(self):
        td = timedelta(seconds=123)
        self.assertEqual(get_total_units(td, TimeUnit.SECOND), 123)

    def test_minute_(self):
        td = timedelta(minutes=2)
        self.assertEqual(get_total_units(td, TimeUnit.MINUTE), 2)

    def test_hour_(self):
        td = timedelta(hours=1)
        self.assertEqual(get_total_units(td, TimeUnit.HOUR), 1)

    def test_day_(self):
        td = timedelta(days=3)
        self.assertEqual(get_total_units(td, TimeUnit.DAY), 3)

    def test_year_(self):
        td = timedelta(days=365.25)
        self.assertAlmostEqual(get_total_units(td, TimeUnit.YEAR), 1.0, places=2)

    def test_invalid_unit_(self):
        td = timedelta(seconds=123)
        with self.assertRaises(ValueError):
            get_total_units(td, "invalid_unit")

    def test_year__(self):
        diff = timedelta(days=365)
        expected_result = (1, TimeUnit.YEAR)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_day__(self):
        diff = timedelta(days=2)
        expected_result = (2, TimeUnit.DAY)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_hour__(self):
        diff = timedelta(hours=1)
        expected_result = (1, TimeUnit.HOUR)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_minute__(self):
        diff = timedelta(minutes=2)
        expected_result = (2, TimeUnit.MINUTE)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_second__(self):
        diff = timedelta(seconds=30)
        expected_result = (30, TimeUnit.SECOND)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_millisecond__(self):
        diff = timedelta(milliseconds=500)
        expected_result = (500, TimeUnit.MILLISECOND)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_microsecond__(self):
        diff = timedelta(microseconds=200)
        expected_result = (200, TimeUnit.MICROSECOND)
        self.assertEqual(check_time_unit(diff), expected_result)

    def test_invalid_time(self):
        diff = timedelta(microseconds=0)
        with self.assertRaises(ValueError):
            check_time_unit(diff)

    def test_convert_datetime_index(self):
        # Create a DataFrame with a datetime index
        data = {'value': [1, 2, 3, 4, 5]}
        index = pd.date_range(start='2023-01-01', periods=5)
        dataset = pd.DataFrame(data, index=index)

        # Call the function to convert the index to datetime
        convert_dataset_index_to_datetime(dataset)

        # Check if the index is in datetime format
        self.assertTrue(is_datetime64_any_dtype(dataset.index))

    def test_non_datetime_index(self):
        # Create a DataFrame with a non-datetime index
        data = {'value': [1, 2, 3, 4, 5]}
        index = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        dataset = pd.DataFrame(data, index=index)

        # Call the function to convert the index to datetime
        convert_dataset_index_to_datetime(dataset)

        # Check if the index is converted to datetime format
        self.assertTrue(is_datetime64_any_dtype(dataset.index))

    def test_conversion(self):
        self.assertEqual(convert_seconds_to_time_unit(1, TimeUnit.SECOND), 1)
        self.assertEqual(convert_seconds_to_time_unit(60, TimeUnit.MINUTE), 1)
        self.assertEqual(convert_seconds_to_time_unit(3600, TimeUnit.HOUR), 1)


if __name__ == '__main__':
    unittest.main()

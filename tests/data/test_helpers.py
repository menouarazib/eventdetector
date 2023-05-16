import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from sympy.testing import pytest

from eventdetector.data.helpers import sliding_windows, convert_dataframe_to_sliding_windows, compute_middle_event


def test_sliding_windows():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected_output = np.array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]], [[7, 8, 9], [10, 11, 12]]])
    assert np.array_equal(sliding_windows(data, width=2, step=1), expected_output)


class TestHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def test_sliding_windows(self):
        # Test case 1: 1D input
        data1 = np.array([1, 2, 3, 4, 5])
        result1 = sliding_windows(data1, width=3, step=1)
        expected1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.assertTrue(np.array_equal(result1, expected1))

        # Test case 2: window width greater than the size of the input data
        data2 = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            sliding_windows(data2, width=6, step=1)

        # Test case 3: 2D input
        data3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result3 = sliding_windows(data3, width=2, step=1)
        expected3 = np.array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]]])
        assert np.array_equal(result3, expected3)

        # Test case 4: 2D input
        data4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result4 = sliding_windows(data4, width=2, step=2)
        expected4 = np.array([[[1, 2, 3], [4, 5, 6]]])
        assert np.array_equal(result4, expected4)

    def test_convert_dataframe_to_sliding_windows(self):
        # Create a sample DataFrame with datetime index and real-valued features
        n: int = 100
        data = np.random.rand(n, 3)
        index = pd.date_range(start='2022-01-01', periods=n, freq='D')
        df = pd.DataFrame(data=data, columns=['feat1', 'feat2', 'feat3'], index=index)

        # Test sliding window generation with default settings
        sw = convert_dataframe_to_sliding_windows(df, width=2, step=1)
        expected_shape = ((n - 1, 2, 4))  # Number of windows, window width, number of features+time
        self.assertEqual(sw.shape, expected_shape)

        # Test sliding window generation with custom settings
        sw = convert_dataframe_to_sliding_windows(df, width=14, step=7, fill_method='ffill')
        expected_shape = ((13, 14, 4))  # Number of windows, window width, number of features+time
        self.assertEqual(sw.shape, expected_shape)

    def test_compute_middle_event(self):
        # Test case 1: List of events with 2 columns
        events_list = [['2022-01-01', '2022-01-02'], ['2022-01-03', '2022-01-05']]
        expected_output = pd.DataFrame({'Middle Event': [datetime(2022, 1, 1, 12, 0), datetime(2022, 1, 4)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_list)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)

        # Test case 2: List of events with 1 column
        events_list = [['2022-01-01'], ['2022-01-03']]
        expected_output = pd.DataFrame({'Middle Event': [datetime(2022, 1, 1), datetime(2022, 1, 3)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_list)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)
        # Test case 3: Pandas DataFrame with 2 columns
        events_df = pd.DataFrame({'Starting Date': ['2022-01-01', '2022-01-03'],
                                  'Ending Date': ['2022-01-02', '2022-01-05']})
        expected_output = pd.DataFrame({'Middle Event': [datetime(2022, 1, 1, 12, 0), datetime(2022, 1, 4)]})
        # call function to get actual output
        actual_output = compute_middle_event(events_df)

        # compare expected and actual outputs
        pd.testing.assert_frame_equal(expected_output, actual_output)

        # Test case 4: Pandas DataFrame with 1 column
        events_df = pd.DataFrame({'Starting Date': ['2022-01-01', '2022-01-03']})
        expected_output = pd.DataFrame({'Middle Event': [datetime(2022, 1, 1), datetime(2022, 1, 3)]})
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


if __name__ == '__main__':
    unittest.main()

import unittest
from datetime import datetime, timedelta

from eventdetector_ts.data.interval import Interval


class TestInterval(unittest.TestCase):
    def setUp(self):
        self.interval1 = Interval(datetime(2010, 7, 21, 18, 25), datetime(2010, 7, 21, 18, 28))
        self.interval2 = Interval(datetime(2010, 7, 21, 18, 24, 30), datetime(2010, 7, 21, 18, 27, 30))
        self.interval3 = Interval(datetime(2010, 7, 21, 18, 26, 30), datetime(2010, 7, 21, 18, 29, 30))

    def test_overlap(self):
        self.assertEqual(self.interval1.overlap(self.interval2), timedelta(seconds=150))
        self.assertEqual(self.interval1.overlap(self.interval3), timedelta(seconds=90))
        self.assertEqual(self.interval2.overlap(self.interval3), timedelta(seconds=60))

    def test_overlapping_parameter(self):
        self.assertEqual(round(self.interval1.overlapping_parameter(self.interval2), 3), 0.714)
        self.assertEqual(round(self.interval1.overlapping_parameter(self.interval3), 3), 0.333)
        self.assertEqual(round(self.interval2.overlapping_parameter(self.interval3), 3), 0.200)


if __name__ == '__main__':
    unittest.main()

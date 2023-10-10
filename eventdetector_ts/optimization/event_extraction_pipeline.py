from datetime import timedelta
from math import ceil
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from eventdetector_ts import MIDDLE_EVENT_LABEL, TimeUnit, config_dict
from eventdetector_ts.data.helpers_data import get_timedelta, get_total_units
from eventdetector_ts.optimization import logger
from eventdetector_ts.optimization.algorithms import convolve_with_gaussian_kernel


class OptimizationData:
    """
    OptimizationData class represents the data used for the event extraction pipeline.

     Attributes:
        - time_unit (TimeUnit): Unit of time used in the dataset.
        - true_events (pd.DataFrame): DataFrame to store true events.
        - predicted_op (np.ndarray): Array to store predicted outcomes.
        - delta Union[int, float]: The maximum time tolerance used to determine the correspondence between a predicted 
            event and its actual counterpart in the true events.
        - s_h (float): A step parameter for the peak height threshold h.
        - s_s (int): Step size in time unit for overlapping the partition.
        - w_s (int): Size in time unit of the overlapping partition.
        - t_max (float): The maximum total time related to sigma.
        - output_dir (str): The parent directory.
        - big_sigma (int): Value calculated based on t_max, w_s, and s_s.
        - overlapping_partitions (np.ndarray): Array to store overalapping partitions.

    """

    def __init__(self, t_max: float, w_s: int, s_s: int,
                 s_h: float,
                 delta: Union[int, float],
                 output_dir: str, time_unit: TimeUnit):
        """
        Initializes the OptimizationData object.

        Args:
            t_max (float): The maximum total time related to sigma.
            w_s (int): Size in time unit of the overalapping partition.
            s_s (int): Step size in time unit for overalapping the partition.
            s_h (float): A step parameter for the peak height threshold h.
            delta Union[int, float]: The maximum time tolerance used to determine the correspondence between a predicted
                event and its actual counterpart in the true events.
            output_dir (str): The parent directory.
            time_unit (TimeUnit): Unit of time used in the dataset.
        """
        self.time_unit = time_unit
        self.true_events: pd.DataFrame = pd.DataFrame()
        self.predicted_op: np.ndarray = np.empty(shape=(0,))
        self.delta = delta
        self.s_h = s_h
        self.s_s = s_s
        self.w_s = w_s
        self.t_max = t_max
        self.output_dir = output_dir
        self.big_sigma = 1 + ceil((self.t_max - self.w_s) / self.s_s)
        self.overlapping_partitions: np.ndarray = np.empty(shape=(0,))

    def set_true_events(self, true_events: pd.DataFrame) -> None:
        self.true_events = true_events

    def set_overlapping_partitions(self, overlapping_partitions: np.ndarray):
        self.overlapping_partitions = overlapping_partitions

    def set_predicted_op(self, predicted_op: np.ndarray):
        self.predicted_op = predicted_op
        overlapping_partitions_test = self.overlapping_partitions[-len(predicted_op):]
        self.overlapping_partitions = overlapping_partitions_test
        first_partition_test_data = self.overlapping_partitions[0]
        last_partition_test_data = self.overlapping_partitions[-1]
        start_date_test_data = first_partition_test_data[0][-1].to_pydatetime()
        end_date_test_data = last_partition_test_data[0][-1].to_pydatetime()
        logger.info(
            f"Starting and ending dates of test data are respectively {start_date_test_data} --> {end_date_test_data}")

        true_events_test = self.true_events[(self.true_events[MIDDLE_EVENT_LABEL] >= start_date_test_data) & (
                self.true_events[MIDDLE_EVENT_LABEL] <= end_date_test_data)]
        self.true_events = true_events_test


def get_peaks(h: float, t: np.ndarray, op_g: np.ndarray) -> np.ndarray:
    """
    Compute peaks for given mid_times of partitions, op values, and threshold h.
    Args:
        h (float): Threshold for peaks.
        t (np.ndarray): mid_times of partitions
        op_g (np.ndarray): op values

    Returns:
        np.ndarray: Peaks.
    """
    peaks, _ = find_peaks(op_g, height=np.array([h, 1.0]))
    return t[peaks]


def compute_op_as_mid_times(overlapping_partitions: np.ndarray, op_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute op as a function of mid-times of partitions instead of partition's index.
    Args:
        overlapping_partitions (np.ndarray): overalapping partitions
        op_g (np.ndarray): Op array

    Returns:
        Tuple[np.ndarray, np.ndarray]: mid-times of partitions, op as a function of mid-times of partitions
    """
    t = []
    op_g_ = []
    for n in range(len(op_g)):
        w_n = overlapping_partitions[n]
        b_n = w_n[0][-1].to_pydatetime()
        e_n = w_n[-1][-1].to_pydatetime()
        c_n = b_n + (e_n - b_n) / 2
        t.append(c_n)
        op_g_.append(op_g[n])
    t, op_g_ = np.array(t), np.array(op_g_)
    return t, op_g_


class OptimizationCalculator:
    def __init__(self, optimization_data: OptimizationData):
        self.optimization_data = optimization_data

    def apply_gaussian_filter(self, sigma: int, m: int) -> np.ndarray:
        return convolve_with_gaussian_kernel(self.optimization_data.predicted_op, sigma, m=m)

    def __compute_op_as_mid_times(self, op_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return compute_op_as_mid_times(self.optimization_data.overlapping_partitions, op_g)

    def __util_method(self, s_peaks: np.ndarray, delta_with_time_unit: timedelta) -> Tuple[int, int, int, list]:
        """
        Useful method for compute_f1score method.
        Args:
            s_peaks (np.ndarray): peaks of op.
            delta_with_time_unit (timedelta): delta as number in unit time.

        Returns:
            tp, fp, fn, delta_t
        """
        e_t = self.optimization_data.true_events.copy()

        fp: int = 0
        tp: int = 0
        delta_t: list = []
        for m_p in s_peaks:
            signed_delta = delta_with_time_unit
            t_t = None
            for i, t_e in enumerate(e_t[MIDDLE_EVENT_LABEL]):
                m_t = t_e
                diff = m_p - m_t

                if abs(diff) <= delta_with_time_unit:
                    if t_t is None or abs(m_p - t_t) > abs(diff):
                        t_t = m_t
                        signed_delta = diff

            if t_t is not None:
                tp += 1
                e_t = e_t.drop(e_t[e_t[MIDDLE_EVENT_LABEL] == t_t].index)
                diff = get_total_units(timedelta_=signed_delta, unit=self.optimization_data.time_unit)

                delta_t.append(diff)
            else:
                fp += 1
        fn: int = len(e_t)
        return tp, fp, fn, delta_t

    def compute_f1score(self, sigma: int, m: int, h: float):
        delta_with_time_unit = get_timedelta(self.optimization_data.delta, self.optimization_data.time_unit)
        op_g: np.ndarray = self.apply_gaussian_filter(sigma=sigma, m=m)
        t, op_g = self.__compute_op_as_mid_times(op_g=op_g)
        s_peaks = get_peaks(h=h, t=t, op_g=op_g)
        tp, fp, fn, delta_t = self.__util_method(s_peaks=s_peaks, delta_with_time_unit=delta_with_time_unit)

        if tp + fp == 0 or tp + fn == 0:
            return 0.0, 0.0, 0.0, [], []

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0, 0.0, 0.0, [], []
        return (2.0 * precision * recall) / (precision + recall), precision, recall, s_peaks.tolist(), delta_t

    def evaluate_combination(self, combination):
        sigma, m, h = combination
        f1_score, precision, recall, peaks, delta_t = self.compute_f1score(sigma, m, h)
        formatted_combination = ', '.join(f'{item:.4f}' for item in combination)
        if f1_score > 0:
            logger.info(
                f"Evaluated Combination [sigma, m, h] : [{formatted_combination}] => [F1 Score: {f1_score:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}]")
        return f1_score, precision, recall, peaks, delta_t


class EventOptimization:
    """
    After obtaining the predicted op values from the metamodel, they are then processed
        through an optimization algorithm to extract the predicted events. This involves applying
        a Gaussian filter to smooth out the predictions and identifying peaks in the resulting signal
        that correspond to the mid-times of the predicted events, which are then compared to the
        actual events in the test set. The performance of the algorithm is evaluated by computing
        metrics such as F1-Score, which combines precision and recall using their harmonic means.
        Maximizing the F1-Score is the preferred metric for evaluating models since it requires
        simultaneously maximizing precision and recall.
    """

    def __init__(self, optimization_data: OptimizationData):
        self.optimization_data = optimization_data
        self.optimization_calculator: OptimizationCalculator = OptimizationCalculator(self.optimization_data)
        self.results = ()

    def max_f1score(self) -> tuple[list, list]:
        """
        The optimization process aims to maximize the F1-Score metric by fine-tuning several parameters,
            including the filter size (2m + 1) and standard deviation (σ) of the Gaussian filter,
            and the peak height threshold h.

        Returns:
              list of peaks, delta_t
        """
        sigma_range = range(1, self.optimization_data.big_sigma + 1)
        h_values = np.arange(0, 1, self.optimization_data.s_h)
        # Create a list of all combinations to evaluate
        combinations = [(sigma, m, h) for sigma in sigma_range for m in [sigma, 2 * sigma, 3 * sigma] for
                        h in h_values]

        try:
            # Evaluate combinations sequentially
            results = [self.optimization_calculator.evaluate_combination(combination) for combination in
                       combinations]
        except ValueError as e:
            logger.error(e)
            exit(0)

        # Find the combination with the maximum F1 score
        best_combination_index = np.argmax(list(map(lambda metrics: metrics[0], results)))
        best_combination = combinations[best_combination_index]
        config_dict["best_combination"] = best_combination
        self.results = results[best_combination_index]
        max_f1_score, precision, recall, peaks, delta_t = self.results

        formatted_combination = ', '.join(f'{item:.4f}' for item in best_combination)
        logger.warning(
            f"Best Combination [sigma, m, h] : [{formatted_combination}] => "
            f"[Max F1 Score: {max_f1_score:.4f} => Precision:{precision:.4f}, Recall:{recall:.4f}]")
        return peaks, delta_t

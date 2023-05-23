import multiprocessing as mp
from math import ceil
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from eventdetector import MIDDLE_EVENT_LABEL, TimeUnit, config_dict
from eventdetector.data.helpers import get_timedelta, get_total_units
from eventdetector.optimization import logger
from eventdetector.optimization.algorithms import convolve_with_gaussian_kernel


class OptimizationData:
    """
    OptimizationData class represents the data used for the event extraction pipeline.

     Attributes:
        - time_unit (TimeUnit): Unit of time used in the dataset.
        - true_events (pd.DataFrame): DataFrame to store true events.
        - predicted_op (np.ndarray): Array to store predicted outcomes.
        - delta (int): The maximum time tolerance used to determine the correspondence between a predicted event
                    and its actual counterpart in the true events.
        - s_h (float): A step parameter for the peak height threshold h.
        - s_s (int): Step size in time unit for sliding the window.
        - w_s (int): Size in time unit of the sliding window.
        - t_max (float): The maximum total time related to sigma.
        - output_dir (str): The parent directory.
        - big_sigma (int): Value calculated based on t_max, w_s, and s_s.
        - sliding_windows (np.ndarray): Array to store sliding windows.

    """

    def __init__(self, t_max: float, w_s: int, s_s: int,
                 s_h: float,
                 delta: int,
                 output_dir: str, time_unit: TimeUnit):
        """
        Initializes the OptimizationData object.

        Args:
            t_max (float): The maximum total time related to sigma.
            w_s (int): Size in time unit of the sliding window.
            s_s (int): Step size in time unit for sliding the window.
            s_h (float): A step parameter for the peak height threshold h.
            delta (int): The maximum time tolerance used to determine the correspondence between a predicted event
                    and its actual counterpart in the true events.
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
        self.sliding_windows: np.ndarray = np.empty(shape=(0,))

    def set_true_events(self, true_events: pd.DataFrame) -> None:
        self.true_events = true_events

    def set_sliding_windows(self, sliding_windows: np.ndarray):
        self.sliding_windows = sliding_windows

    def set_predicted_op(self, predicted_op: np.ndarray):
        self.predicted_op = predicted_op
        sliding_windows_test = self.sliding_windows[-len(predicted_op):]
        self.sliding_windows = sliding_windows_test
        first_window_test_data = self.sliding_windows[0]
        last_window_test_data = self.sliding_windows[-1]
        start_date_test_data = first_window_test_data[0][-1].to_pydatetime()
        end_date_test_data = last_window_test_data[0][-1].to_pydatetime()
        logger.info(
            f"Starting and ending dates of test data are respectively {start_date_test_data} --> {end_date_test_data}")

        true_events_test = self.true_events[(self.true_events[MIDDLE_EVENT_LABEL] >= start_date_test_data) & (
                self.true_events[MIDDLE_EVENT_LABEL] <= end_date_test_data)]
        self.true_events = true_events_test


def get_peaks(h, t: np.ndarray, op_g: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(op_g, height=np.array([h, 1.0]))
    return t[peaks]


class OptimizationCalculator:
    def __init__(self, optimization_data: OptimizationData):
        self.optimization_data = optimization_data

    def apply_gaussian_filter(self, sigma: int, m: int) -> np.ndarray:
        return convolve_with_gaussian_kernel(self.optimization_data.predicted_op, sigma, m=m)

    def __compute_op_as_mid_times(self, op_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = []
        op_g_ = []
        for n in range(len(op_g)):
            w_n = self.optimization_data.sliding_windows[n]
            b_n = w_n[0][-1].to_pydatetime()
            e_n = w_n[-1][-1].to_pydatetime()
            c_n = b_n + (e_n - b_n) / 2
            t.append(c_n)
            op_g_.append(op_g[n])
        t, op_g_ = np.array(t), np.array(op_g_)
        return t, op_g_

    def compute_f1score(self, sigma: int, m: int, h: float):
        delta_with_time_unit = get_timedelta(self.optimization_data.delta, self.optimization_data.time_unit)
        op_g: np.ndarray = self.apply_gaussian_filter(sigma=sigma, m=m)
        t, op_g = self.__compute_op_as_mid_times(op_g=op_g)
        s_peaks = get_peaks(h=h, t=t, op_g=op_g)
        tp, fp, fn = 0, 0, 0
        delta_t = []
        e_t = self.optimization_data.true_events.copy()
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
                delta_t.append(get_total_units(timedelta_=signed_delta, unit=self.optimization_data.time_unit))
            else:
                fp += 1
        fn = fn + len(e_t)
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
        formatted_combination = ', '.join(f'{item:.2f}' for item in combination)
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

    def max_f1score(self, use_multiprocessing: bool = False) -> tuple[list, list]:
        """
        The optimization process aims to maximize the F1-Score metric by fine-tuning several parameters,
            including the filter size (2m + 1) and standard deviation (σ) of the Gaussian filter,
            and the peak height threshold h.

        Args:
            use_multiprocessing (bool): Using or not multiprocessing

        Returns:
              list of peaks
        """
        sigma_range = range(1, self.optimization_data.big_sigma + 1)
        h_values = np.arange(0, 1, self.optimization_data.s_h)
        # Create a list of all combinations to evaluate
        combinations = [(sigma, m, h) for sigma in sigma_range for m in [sigma, 2 * sigma, 3 * sigma] for
                        h in h_values]

        try:
            if use_multiprocessing:
                # Create a multiprocessing pool with the desired number of processes
                num_processes = mp.cpu_count() // 2  # Use the number of available CPU cores
                pool = mp.Pool(processes=num_processes)
                # Evaluate combinations in parallel
                results = pool.map(self.optimization_calculator.evaluate_combination, combinations)
            else:
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
        max_f1_score, precision, recall, peaks, delta_t = results[best_combination_index]

        formatted_combination = ', '.join(f'{item:.2f}' for item in best_combination)
        logger.info(
            f"Best Combination [sigma, m, h] : [{formatted_combination}] => "
            f"[Max F1 Score: {max_f1_score:.4f} => Precision:{precision:.4f}, Recall:{recall:.4f}]")
        return peaks, delta_t

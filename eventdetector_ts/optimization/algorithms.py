from typing import Union

import numpy as np


def convolve_with_gaussian_kernel(signal: np.ndarray, sigma: Union[int, float], m: int) -> np.ndarray:
    """
    Convolve a signal with a Gaussian kernel.

    Args:
        signal (np.ndarray): The input signal to convolve.
        sigma (Union[int, float]): The standard deviation of the Gaussian kernel.
        m (int): The radius of the kernel.

    Returns:
        np.ndarray: The convolved signal.

    """

    # Create the Gaussian kernel
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(np.arange(-m, m + 1) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Perform the convolution
    convolved_signal = np.convolve(signal, kernel, mode='same')

    return convolved_signal

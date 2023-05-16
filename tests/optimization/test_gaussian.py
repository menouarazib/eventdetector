import unittest

import numpy as np

from eventdetector.optimization.algorithms import convolve_with_gaussian_kernel


def convolution_with_gaussian(signal, sigma, m):
    signal_size = len(signal)

    output = []
    for n in range(signal_size):
        temp = 0
        sum_kernel = 0
        for i in range(-m, m + 1):
            g_i = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(i ** 2) / (2. * sigma ** 2))
            if 0 <= (n - i) < signal_size:
                temp += g_i * signal[n - i]
            sum_kernel += g_i

        output.append(temp / sum_kernel)
    return output


class TestGaussianFilter(unittest.TestCase):
    def test_gaussian_filter(self):
        signal = np.array([1.0, 2, 3, 4.0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        sigma = 1
        m = 2

        convolved_signal = convolve_with_gaussian_kernel(signal=signal, sigma=sigma, m=m)
        convolved_signal_expected = convolution_with_gaussian(signal=signal, sigma=sigma, m=m)

        # Check if the outputs are equal
        np.testing.assert_allclose(convolved_signal_expected, convolved_signal, atol=1e-8)


if __name__ == '__main__':
    unittest.main()

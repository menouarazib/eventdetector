import unittest

import numpy as np
import tensorflow as tf

from eventdetector_ts.models.helpers import CustomEarlyStopping


class TestHelpers(unittest.TestCase):
    def setUp(self):
        pass

    class TestCustomEarlyStopping(tf.test.TestCase):
        def test_on_epoch_end(self):
            # Create a custom early stopping callback
            early_stopping = CustomEarlyStopping(ratio=2.0, patience=3, verbose=0)

            # Set up test data
            x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y_train = np.array([0, 1, 1, 0])
            x_val = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y_val = np.array([0, 1, 1, 0])

            # Define a simple model
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model with the custom early stopping callback
            model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping])

            # Check that training was stopped early
            self.assertLess(early_stopping.stopped_epoch, 10)


if __name__ == '__main__':
    unittest.main()

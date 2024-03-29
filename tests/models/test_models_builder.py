import unittest

import tensorflow as tf
from sympy.testing import pytest

from eventdetector_ts import RNN_ENCODER_DECODER, FFN, CNN, RNN_BIDIRECTIONAL, CONV_LSTM1D, LSTM, SELF_ATTENTION
from eventdetector_ts.models.models_builder import ModelBuilder, ModelCreator


class TestModelsBuilder(unittest.TestCase):
    def setUp(self):
        # create a model builder with an input layer
        self.inputs = tf.keras.layers.Input(shape=(10,))
        self.model_builder = ModelBuilder(self.inputs)
        self.inputs_rnn = tf.keras.Input(shape=(45, 5), name="Input")

    def test_check_input_shape(self):
        # create a layer with compatible input shape and call __check_input_shape
        layer1 = tf.keras.layers.Dense(5)
        output1 = self.model_builder._ModelBuilder__check_input_shape(layer1)
        self.assertEqual(output1.shape, tf.TensorShape([None, 10]))

    def test_add_layer(self):
        layer1 = tf.keras.layers.Dense(5)
        self.model_builder._ModelBuilder__add_layer(layer1)
        self.assertEqual(self.model_builder.outputs.shape, tf.TensorShape([None, 5]))

        layer2 = tf.keras.layers.Conv2D(32, kernel_size=3)
        with pytest.raises(ValueError):
            self.model_builder._ModelBuilder__add_layer(layer2)

    def test_create_models(self):
        model_creator = ModelCreator(
            [(RNN_ENCODER_DECODER, 1), (FFN, 2), (CNN, 2), (RNN_BIDIRECTIONAL, 1), (CONV_LSTM1D, 1), (LSTM, 3),
             (SELF_ATTENTION, 3)],
            hyperparams_rnn=(1, 2, 45, 46, "tanh"),
            hyperparams_cnn=(64, 65, 3, 4, 1, 1, "relu"),
            hyperparams_ffn=(1, 2, 64, 128, "sigmoid"), save_models_as_dot_format=False, root_dir=None, dropout=0.3,
            last_act_func="sigmoid", hyperparams_transformer=(256, 4, 1, True, "relu"))

        model_creator.create_models(inputs=self.inputs_rnn)

        for key, value in model_creator.created_models.items():
            keras_model: tf.keras.Model = value
            self.assertEqual(keras_model.layers[-1].output_shape, (None, 1))


if __name__ == '__main__':
    unittest.main()

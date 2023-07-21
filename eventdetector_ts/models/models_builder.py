import os
import random
from typing import Optional, Tuple, Dict, List

import numpy as np
import tensorflow as tf

from eventdetector_ts import LSTM, GRU, CNN, RNN_BIDIRECTIONAL, RNN_ENCODER_DECODER, CNN_RNN, FFN, CONV_LSTM1D, \
    SELF_ATTENTION, MODELS_DIR
from eventdetector_ts.models import logger_models
from eventdetector_ts.models.helpers import SelfAttention


class DtwLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int = 32):
        super(DtwLoss, self).__init__()
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        tmp = []
        for item in range(self.batch_size):
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = []
            for i in range(n + 1):
                line = []
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        line.append(0)
                    else:
                        line.append(np.inf)
                dtw_matrix.append(line)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]])
                    dtw_matrix[i][j] = tf.cast(cost, dtype=tf.float32) + tf.cast(last_min, dtype=tf.float32)

            temp = []
            for i in range(len(dtw_matrix)):
                temp.append(tf.stack(dtw_matrix[i]))

            tmp.append(tf.stack(temp)[n, m])
        return tf.reduce_mean(tmp)


class ModelBuilder:
    """
    Helper class for building TensorFlow Keras models.
    """

    def __init__(self, inputs):
        """
        Initialize the ModelBuilder object.

        Args:
            inputs (tf.keras.layers.Layer): The input layer of the model.
        """
        self.outputs = None
        self.inputs = inputs

    def __return_input(self):
        """
        Return the input layer of the model if output is None, otherwise return the output layer.

        Returns:
            tf.keras.layers.Layer: The input or output layer of the model.
        """
        if self.outputs is None:
            return self.inputs
        return self.outputs

    def __check_input_shape(self, layer: tf.keras.layers.Layer) -> tf.Tensor:
        """
        Check if the input shape of a layer is compatible with the output shape of the previous layer.

        Example:
            input_1 = tf.keras.layers.Input(shape=(10,))
            input_2 = tf.keras.layers.Input(shape=(20,))
            concat_layer = tf.keras.layers.Concatenate()([input_1, input_2])
            print(concat_layer.input_spec) -> [InputSpec(shape=(None, 10), dtype=tf.float32),
                                                InputSpec(shape=(None, 20), dtype=tf.float32)]

        Args:
            layer: A `tf.keras.layers.Layer` object for which to check the input shape.

        Returns:
            A `tf.Tensor` object representing the output of the previous layer (i.e., the input to the current layer).

        Raises:
            ValueError: If the input shape is not compatible with the output shape of the previous layer.
        """
        input_ = self.__return_input()

        if not hasattr(layer, 'input_spec') or layer.input_spec is None:
            return input_

        input_spec = layer.input_spec
        if not isinstance(input_spec, list):
            input_spec = [input_spec]

        for spec in input_spec:
            # Retrieve the expected shape and dtype from the input spec
            expected_shape = spec.shape
            expected_dtype = spec.dtype

            # Check if the expected shape is defined
            if expected_shape is not None:
                expected_rank = len(expected_shape)
            else:
                # If the expected shape is not defined, set the expected rank to None
                expected_rank = None

            # Get the actual input shape, dtype, and rank
            actual_shape = input_.shape[1:]
            actual_dtype = input_.dtype
            actual_rank = len(actual_shape)

            # Check if the shape, dtype, and rank are compatible
            if expected_shape is not None and expected_shape != actual_shape:
                msg = f"The expected input shape for layer '{layer.name}' is {expected_shape}, " \
                      f"but received {actual_shape} instead."
                logger_models.critical(msg)
                raise ValueError(msg)
            if expected_dtype is not None and expected_dtype != actual_dtype:
                msg = f"The expected input dtype for layer '{layer.name}' is {expected_dtype}, " \
                      f"but received {actual_dtype} instead."
                logger_models.critical(msg)
                raise ValueError(msg)
            if expected_rank is not None and expected_rank != actual_rank:
                msg = f"The expected input rank for layer '{layer.name}' is {expected_rank}, " \
                      f"but received {actual_rank} instead."
                logger_models.critical(msg)
                raise ValueError(msg)

        return input_

    def __add_layer(self, layer: tf.keras.layers.Layer, check_shape: bool = True) -> None:
        """
        Adds a layer to the model and checks if the layer's input shape is compatible with the previous
        layer's output shape.

        Args:
            layer: The layer to be added.
            check_shape: A boolean indicating whether to check the input shape compatibility or not.

        Returns:
            None.

        Raises:
            ValueError: If the input shape is not compatible with the output shape of the previous layer.
        """

        if check_shape:
            input_ = self.__check_input_shape(layer)
        else:
            input_ = self.__return_input()

        # Add the layer to the model by passing its input as input_ and saving the output to self.outputs
        self.outputs = layer(input_)

    def add_lstm_layer(self, hidden_dim: int, activation: str = "tanh", return_sequences: bool = False,
                       dropout: float = 0.3) -> None:
        """
        Adds a Long-Short Term Memory (LSTM) layer to the model.

        Args:
            hidden_dim (int): The number of output units in the LSTM layer.
            activation (str, optional): The activation function to use. Defaults to 'tanh'.
            return_sequences (bool, optional): Whether to return the last output in the output sequence
                or the full sequence. Default to False.
            dropout (float, optional): The dropout rate to use. Default to 0.3.

        Returns:
            None
        """

        lstm = tf.keras.layers.LSTM(units=hidden_dim, activation=activation, return_sequences=return_sequences,
                                    dropout=dropout)
        self.__add_layer(lstm)

    def add_conv1d_layer(self, filters: int, kernel_size: int, strides: int = 1, padding: str = 'same',
                         activation: str = "relu") -> None:
        """
        Adds a 1D convolutional layer to the model.

        Args:
            filters (int): The number of output filters in the convolution.
            kernel_size (int): The size of the convolutional kernel.
            strides (int): The stride of the convolution. Default is 1.
            padding (str): One of 'valid' or 'same'. Default is 'same'.
            activation (str): The activation function to use. Default is 'relu'.

        Returns:
            None
        """

        conv1d_layer = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                              padding=padding, activation=activation)
        self.__add_layer(conv1d_layer)

    def add_dense_layer(self, units: int, activation: str = 'sigmoid', dropout: Optional[float] = 0.3) -> None:
        """
        Add a dense layer to the neural network.

        Args:
            units (int): The number of units in the dense layer.
            activation (str): The activation function to use in the dense layer. Default is 'sigmoid'.
            dropout (Optional[float]): The dropout rate to use in the dense layer. If set to None, no dropout
                layer will be added. Default is 0.3.

        Returns:
            None
        """

        dense_layer = tf.keras.layers.Dense(units=units, activation=activation)
        self.__add_layer(dense_layer)
        if dropout is not None:
            dropout_layer = tf.keras.layers.Dropout(rate=dropout)
            self.__add_layer(dropout_layer)

    def add_gru_layer(self, hidden_dim: int, activation: str = "tanh", return_sequences: bool = False,
                      dropout: float = 0.3) -> None:
        """
        Adds a Gated Recurrent Unit (GRU) layer to the model.

        Args:
            hidden_dim (int): The number of output units in the GRU layer.
            activation (str, optional): The activation function to use. Defaults to 'tanh'.
            return_sequences (bool, optional): Whether to return the last output in the output sequence
                or the full sequence. Defaults to False.
            dropout (float, optional): The dropout rate to use. Default to 0.3.

        Returns:
            None
        """

        gru_layer = tf.keras.layers.GRU(units=hidden_dim, activation=activation, return_sequences=return_sequences,
                                        dropout=dropout)
        self.__add_layer(gru_layer)

    def add_bidirectional(self, hidden_dim: int, activation: str = "tanh",
                          return_sequences: bool = False, dropout: float = 0.3) -> None:
        """
        Adds a Bidirectional LSTM layer to the model.

        Args:
            hidden_dim (int): Number of LSTM units in the layer.
            activation (str, optional): Activation function to use. Defaults to 'tanh'.
            return_sequences (bool, optional): Whether to return the last output in the output sequence,
                                                or the full sequence. Defaults to False.
            dropout (float, optional): The dropout rate to apply. Default to 0.3.

        Returns:
            None
        """

        lstm_fw = tf.keras.layers.LSTM(units=hidden_dim, activation=activation,
                                       return_sequences=return_sequences, dropout=dropout)
        lstm_bw = tf.keras.layers.LSTM(units=hidden_dim, activation=activation,
                                       return_sequences=return_sequences, dropout=dropout, go_backwards=True)
        bidirectional = tf.keras.layers.Bidirectional(lstm_fw, backward_layer=lstm_bw, merge_mode="mul")
        self.__add_layer(bidirectional)

    def add_conv_lstm1d(self, filters: int, kernel_size: int, activation: str = 'tanh',
                        strides: int = 1, padding: str = 'same', dropout: float = 0.3,
                        return_sequences: bool = False) -> None:
        """
        Adds a ConvLSTM1D layer to the neural network.

        Args:
            filters (int): Number of output filters in the convolution.
            kernel_size (int): Size of the 1D convolutional partition.
            activation (str): Activation function to use. Defaults to 'tanh'.
            strides (int): Stride length of the convolution. Default to 1.
            padding (str): Type of padding to use. Defaults to 'same'.
            dropout (float): Fraction of the input units to drop. Default to 0.3.
            return_sequences (bool, optional): Whether to return the last output in the output sequence,
                                                or the full sequence. Defaults to False.

        Returns:
            None
        """

        conv_lstm1d_layer = tf.keras.layers.ConvLSTM1D(filters=filters, kernel_size=kernel_size, activation=activation,
                                                       strides=strides, padding=padding, dropout=dropout,
                                                       return_sequences=return_sequences)
        self.__add_layer(conv_lstm1d_layer)

    def add_flatten_layer(self) -> None:
        """
        Add flatten layer

        Returns:
            None
        """

        flatten = tf.keras.layers.Flatten()
        self.__add_layer(flatten)

    def add_max_pooling1d(self, pool_size: int, strides: Optional[int] = None) -> None:
        """
        Adds a 1D max pooling layer to the model.

        Args:
            pool_size (int): Size of the max pooling partitions.
            strides (int, optional): Factor by which to downscale. Defaults to None.

        Returns:
            None
        """

        # Create a 1D max pooling layer with the given pool_size and strides
        pooling = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides)
        # Add the max pooling layer to the model
        self.__add_layer(pooling)

    def add_reshape(self, shape: tuple) -> None:
        """
        Adds a reshape layer to the model.

        Args:
            shape (tuple): A tuple representing the new shape for the input tensor.

        Returns:
            None
        """

        # Create a reshaped layer with the given shape
        reshape = tf.keras.layers.Reshape(target_shape=shape)
        # Add the reshaped layer to the model
        self.__add_layer(reshape)

    def add_self_attention(self, units: int) -> None:
        """
        Adds a self-attention layer to the model.

        Args:
            units (int): Dimensionality of the output space.

        Returns:
            None
        """

        # Create a self-attention layer with the given units
        self_attention_layer = SelfAttention(units)
        # Add the self-attention layer to the model
        self.__add_layer(self_attention_layer)

    def add_global_max_pooling(self):
        """
        Adds a global max pooling layer to the model.

        Returns:
            None
        """
        # Create a global max pooling layer
        pooling = tf.keras.layers.GlobalMaxPooling1D()
        # Add the global max pooling layer to the model
        self.__add_layer(pooling)

    def build(self, name: str, save_models_as_dot_format: bool, root_dir: Optional[str] = None,
              ) -> tf.keras.Model:
        """
        Builds the model and compiles it.

        Args:
            name: The name of the model.
            save_models_as_dot_format: Whether to save the model as a dot format file.
            root_dir (Optional[str]): The root directory in which to save the model as a dot format file.

        Returns:
            The compiled model.
        """
        model_ = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=name)

        if save_models_as_dot_format and root_dir is not None:
            png_path = os.path.join(root_dir, f"{name}.png")
            tf.keras.utils.plot_model(model_, png_path, show_shapes=True, show_dtype=True,
                                      show_layer_names=True, expand_nested=True)

        model_.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.MeanSquaredError()
        )

        return model_


class ModelCreator:
    """
       A class for creating different neural network models.

       Attributes:
           models (List[Tuple[str, int]]): A list of tuples representing the model types and number of instances.
               Supported model types include LSTM, GRU, CNN, RNN_BIDIRECTIONAL, CONV_LSTM1D, RNN_ENCODER_DECODER,
               CNN_RNN, SELF_ATTENTION, and FFN.
           hyperparams_ffn (Tuple[int, int, int]): Specify the hyper-parameters for the FFN.
           hyperparams_cnn (Tuple[int, int, int, int, int]): Specify the hyper-parameters for the CNN.
           hyperparams_rnn (Tuple[int, int, int]): Specify the hyper-parameters for the RNN.
           inputs (tf.keras.Input): The input layer for the neural network.
           save_models_as_dot_format (bool): Whether to save the models as a dot format file.
           root_dir (str): The root directory where the created models will be saved.

       """

    def __init__(self, models: List[Tuple[str, int]], hyperparams_ffn: Tuple[int, int, int],
                 hyperparams_cnn: Tuple[int, int, int, int, int], hyperparams_rnn: Tuple[int, int, int],
                 save_models_as_dot_format: bool,
                 root_dir: Optional[str] = None):
        """
        Initialize the ModelCreator class with the given arguments.

        Args:
            models (List[Tuple[str, int]]): A list of tuples representing the model types and
                number of instances. Supported model types include LSTM, GRU, CNN, RNN_BIDIRECTIONAL, CONV_LSTM1D,
                RNN_ENCODER_DECODER, CNN_RNN, SELF_ATTENTION, and FFN.
            hyperparams_ffn (Tuple[int, int, int]): Specify the hyper-parameters for the FFN.
            hyperparams_cnn (Tuple[int, int, int, int, int]): Specify the hyper-parameters for the CNN.
            hyperparams_rnn (Tuple[int, int, int]): Specify the hyperparameters for the RNN.
            save_models_as_dot_format (bool): Whether to save the models as a dot format file.
                The default value is False. If set to True, then you should have graphviz software
                to be installed on your machine.
            root_dir (str): The root directory where the created models will be saved.
        """
        self.save_models_as_dot_format = save_models_as_dot_format
        self.train_losses: Dict[str, np.ndarray] = {}
        self.val_losses: Dict[str, np.ndarray] = {}
        self.root_dir: str = root_dir
        self.inputs = None
        self.hyperparams_rnn = hyperparams_rnn
        self.hyperparams_cnn = hyperparams_cnn
        self.hyperparams_ffn = hyperparams_ffn
        self.models = models
        self.created_models: Dict[str, tf.keras.Model] = {}
        self.__create_models_dir()

    def __create_models_dir(self):
        if self.root_dir is not None:
            # Create the directory if it doesn't exist
            models_dir = os.path.join(self.root_dir, MODELS_DIR)
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            self.root_dir = models_dir

    def __create_lstm_networks(self) -> None:
        """
        Creates LSTM networks with random number of layers and units within the specified range.

        Returns:
            None
        """
        # Get the maximum number of LSTM units and layers from hyperparameters
        max_layers, min_units, max_units = self.hyperparams_rnn
        # Get the number of LSTM instances from the model list
        num_instances = self.__get_instances(LSTM)
        # If there are LSTM instances, create them
        if num_instances > 0:
            for i in range(num_instances):
                # Create a new instance of the ModelBuilder class
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                # Set a base name for the model
                name = f"{LSTM}_{i}"
                # Choose a random number of layers between 1 and the maximum number of layers
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    if j == num_layers - 1:
                        model.add_lstm_layer(hidden_dim=units_j)
                    else:
                        model.add_lstm_layer(hidden_dim=units_j, return_sequences=True)
                # adds a dense layer with a single unit to a neural network model,
                # for regression where the output is a continuous numerical value.
                model.add_dense_layer(units=1, dropout=None)
                # Build the model with the chosen name and save it to the created_models dictionary
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_gru_networks(self) -> None:
        """
        Creates GRU networks with random number of layers and units within the specified range.

        Returns:
            None
        """
        max_layers, min_units, max_units = self.hyperparams_rnn
        num_instances = self.__get_instances(GRU)

        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{GRU}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    if j == num_layers - 1:
                        model.add_gru_layer(hidden_dim=units_j)
                    else:
                        model.add_gru_layer(hidden_dim=units_j, return_sequences=True)
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_cnn_networks(self):
        """
        Create convolutional neural network models with a random number of convolutional layers and filters.

        Returns:
            None
        """
        filters_min, filters_max, kernel_size_min, kernel_size_max, max_layers = self.hyperparams_cnn
        num_instances = self.__get_instances(CNN)

        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{CNN}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                filters_per_layer = random.randint(filters_min, filters_max + 1)
                kernel_size_per_layer = random.randint(kernel_size_min, kernel_size_max + 1)
                for j in range(num_layers):
                    filters = max(1, filters_per_layer // (2 ** (num_layers - j - 1)))
                    kernel_size = kernel_size_per_layer
                    model.add_conv1d_layer(filters=filters, kernel_size=kernel_size)
                    if j % 2 == 1:
                        model.add_max_pooling1d(pool_size=2, strides=2)

                # Add flatten layer before dense layer
                model.add_flatten_layer()
                # Add dense output layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_bi_lstm(self):
        max_layers, min_units, max_units = self.hyperparams_rnn
        num_instances = self.__get_instances(RNN_BIDIRECTIONAL)
        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{RNN_BIDIRECTIONAL}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    if j == num_layers - 1:
                        model.add_bidirectional(hidden_dim=units_j)
                    else:
                        model.add_bidirectional(hidden_dim=units_j, return_sequences=True)
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_ffn(self):
        max_layers, min_units, max_units = self.hyperparams_ffn
        num_instances = self.__get_instances(FFN)
        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                model.add_flatten_layer()
                name = f"{FFN}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    model.add_dense_layer(units=units_j)
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_rnn_encoder_decoder(self):
        max_layers, min_units, max_units = self.hyperparams_rnn
        num_instances = self.__get_instances(RNN_ENCODER_DECODER)
        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{RNN_ENCODER_DECODER}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    if j == num_layers - 1:
                        model.add_lstm_layer(hidden_dim=units_j, return_sequences=True)
                        model.add_gru_layer(hidden_dim=units_j)
                    else:
                        model.add_lstm_layer(hidden_dim=units_j, return_sequences=True)
                        model.add_gru_layer(hidden_dim=units_j, return_sequences=True)

                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_cnn_rnn(self):
        max_layers, min_units, max_units = self.hyperparams_rnn
        filters_max, filters_min, kernel_size_min, kernel_size_max, max_layers_ = self.hyperparams_cnn
        max_layers = max(max_layers_, max_layers)
        num_instances = self.__get_instances(CNN_RNN)
        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{CNN_RNN}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                filters_per_layer = random.randint(filters_min, filters_max + 1)
                kernel_size_per_layer = random.randint(kernel_size_min, kernel_size_max + 1)
                for j in range(num_layers):
                    units_j = units[j]
                    filters = max(1, filters_per_layer // (2 ** (num_layers - j - 1)))
                    kernel_size = kernel_size_per_layer
                    model.add_conv1d_layer(filters=filters, kernel_size=kernel_size)
                    model.add_lstm_layer(hidden_dim=units_j, return_sequences=True)
                    if j % 2 == 1:
                        model.add_max_pooling1d(pool_size=2, strides=2)

                # Add flatten layer before dense layer
                model.add_flatten_layer()
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_conv_lstm1d(self):
        filters_min, filters_max, kernel_size_min, kernel_size_max, max_layers = self.hyperparams_cnn
        num_instances = self.__get_instances(CONV_LSTM1D)
        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                shape_inputs = tf.keras.backend.int_shape(model.inputs)
                new_shape = (shape_inputs[1], 1, shape_inputs[2])
                model.add_reshape(shape=new_shape)
                name = f"{CONV_LSTM1D}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                filters_per_layer = random.randint(filters_min, filters_max + 1)
                kernel_size_per_layer = random.randint(kernel_size_min, kernel_size_max + 1)
                for j in range(num_layers):
                    filters = max(1, filters_per_layer // (2 ** (num_layers - j - 1)))
                    kernel_size = kernel_size_per_layer
                    model.add_conv_lstm1d(filters=filters, kernel_size=kernel_size, return_sequences=True)

                # Add flatten layer before dense layer
                model.add_flatten_layer()
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def __create_encoder_decoder_self_attention(self):
        max_layers, min_units, max_units = self.hyperparams_rnn
        num_instances = self.__get_instances(SELF_ATTENTION)

        if num_instances > 0:
            for i in range(num_instances):
                model: ModelBuilder = ModelBuilder(inputs=self.inputs)
                name = f"{SELF_ATTENTION}_{i}"
                num_layers = random.randint(1, max_layers + 1)
                units = [random.randint(min_units, max_units + 1) for _ in range(num_layers)]
                units = sorted(units, reverse=True)
                for j in range(num_layers):
                    units_j = units[j]
                    model.add_bidirectional(hidden_dim=units_j, return_sequences=True)
                    model.add_gru_layer(hidden_dim=units_j, return_sequences=True)
                    model.add_self_attention(units=units_j)
                    if j == num_layers - 1:
                        # Two cases
                        case1 = "USE_LSTM"
                        case2 = "USE_MAX_GLOBAl_POOLING"
                        chosen_case = random.choice([case1, case2])
                        if chosen_case == case1:
                            model.add_lstm_layer(hidden_dim=units_j, return_sequences=False)
                        else:
                            model.add_global_max_pooling()
                # Add last layer for regression
                model.add_dense_layer(units=1, dropout=None)
                keras_model = model.build(name=name, save_models_as_dot_format=self.save_models_as_dot_format,
                                          root_dir=self.root_dir)
                self.created_models[name] = keras_model

    def create_models(self, inputs: tf.keras.Input) -> None:
        """
        Create different neural networks according to {self.models}

        Args:
            inputs: The input layer for the neural network.

        Returns:
            None
        """
        self.inputs = inputs
        # Create LSTM networks
        self.__create_lstm_networks()
        # Create GRU networks
        self.__create_gru_networks()
        # Create feed-forward neural networks
        self.__create_ffn()
        # Create CNN networks
        self.__create_cnn_networks()
        # Create bidirectional LSTM network
        self.__create_bi_lstm()
        # Create CNN-RNN hybrid network
        self.__create_cnn_rnn()
        # Create RNN Encoder-Decoder network
        self.__create_rnn_encoder_decoder()
        # Create Convolutional LSTM network
        self.__create_conv_lstm1d()
        # Create Encoder-Decoder network with a Self-Attention mechanism
        self.__create_encoder_decoder_self_attention()

    def __get_instances(self, model_type: str) -> int:
        # Loop through the model list to find the model of the given type
        for model in self.models:
            if model[0] == model_type:
                # Return the number of instances of the model
                return model[1]

        # Return 0 if no instance of the model type was found
        return 0

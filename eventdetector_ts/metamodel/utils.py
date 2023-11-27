import os
import re
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from eventdetector_ts import MIN_MAX_SCALER, ROBUST_SCALER, SCALERS_DIR, FILL_NAN_ZEROS, FILL_NAN_FFILL, \
    FILL_NAN_BFILL, FILL_NAN_MEDIAN, RNN_BIDIRECTIONAL, CONV_LSTM1D, RNN_ENCODER_DECODER, FFN, CNN_RNN, \
    GRU, CNN, SELF_ATTENTION, LSTM, TYPE_TRAINING_AVERAGE, TYPE_TRAINING_FFN, STANDARD_SCALER, TRANSFORMER
from eventdetector_ts.data.helpers_data import InvalidArgumentError


class DataSplitter:
    """
    A class for splitting and scaling data into training, test sets and applying scalers to each
    time step in the data.
    """

    def __init__(self, test_size: float, scaler_type: str):
        """
        Initialize the DataSplitter object.

        Args:
            test_size: The fraction of data to use for testing.
            scaler_type: The type of scaler to use.
        """

        self.train_x: np.ndarray = np.empty(shape=(0,))
        self.test_x: np.ndarray = np.empty(shape=(0,))
        self.train_y: np.ndarray = np.empty(shape=(0,))
        self.test_y: np.ndarray = np.empty(shape=(0,))
        self.scalers: Dict[int, StandardScaler | MinMaxScaler | ROBUST_SCALER] = {}
        self.test_size: float = test_size
        self.scaler_type: str = scaler_type

    def split_data_and_apply_scaler(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Split the data into training, validation, and test sets and apply the specified scaler to each time step.

        Args:
            x: The input data with shape (n_samples, n_time_steps, n_features).
            y: The target data with shape (n_samples,).

        Returns:
            A tuple containing the training, validation, and test sets as numpy arrays and a dictionary of scalers.
        """
        assert x.ndim == 3, "x must be a 3D array."
        assert y.ndim == 1, "y must be a 1D array."
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples."

        # Split the data into training and test sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=self.test_size,
                                                                                shuffle=False)

        n_time_steps = x.shape[1]

        self.scalers = {}
        # Apply scaler to each time step
        for i in range(n_time_steps):
            scaler = StandardScaler()
            if self.scaler_type == MIN_MAX_SCALER:
                scaler = MinMaxScaler()
            elif self.scaler_type == ROBUST_SCALER:
                scaler = RobustScaler()
            self.scalers[i] = scaler
            self.train_x[:, i, :] = self.scalers[i].fit_transform(self.train_x[:, i, :])
            self.test_x[:, i, :] = self.scalers[i].transform(self.test_x[:, i, :])

    def save_scalers(self, output_dir: str) -> None:
        """
        Saves the scalers to disk.

        Args:
            output_dir: the directory where the scalers should be saved

        Returns:
            None
        """
        # Create the directory if it doesn't exist
        scalers_dir = os.path.join(output_dir, SCALERS_DIR)
        if not os.path.exists(scalers_dir):
            os.makedirs(scalers_dir)

        # Save each scaler to disk
        n_time_steps: int = self.test_x.shape[1]
        for i in range(n_time_steps):
            # Generate the path to save the scaler to
            scaler_i_path = os.path.join(scalers_dir, f'scaler_{i}.joblib')
            # Print progress
            print("\rSaving scaling...{}/{}".format(i + 1, n_time_steps), end="")
            # Save the scaler to disk
            joblib.dump(self.scalers[i], scaler_i_path)
        print()


def validate_required_args(meta_model) -> None:
    """
       Validate the required arguments of the MetaModel.

       Args:
           meta_model (MetaModel): A MetaModel instance.

       Returns:
           None

       Raises:
           ValueError: If any of the arguments are invalid.
       """

    __validate_required_args(meta_model=meta_model)

    if meta_model.dataset is None or meta_model.dataset.empty:
        raise InvalidArgumentError("dataset cannot be None or empty.")
    elif not isinstance(meta_model.dataset, pd.DataFrame):
        raise InvalidArgumentError("dataset should be a Pandas DataFrame.")

    if len(meta_model.dataset) < meta_model.width:
        raise InvalidArgumentError("Dataset length is smaller than the given partition width.")

    if meta_model.events is None or (isinstance(meta_model.events, pd.DataFrame) and meta_model.events.empty) or \
            (isinstance(meta_model.events, list) and len(meta_model.events) == 0):
        raise InvalidArgumentError("Events is empty or None.")
    elif not isinstance(meta_model.events, (list, pd.DataFrame)):
        raise InvalidArgumentError("Events should be a list or a Pandas DataFrame.")

    if not re.match(r"^\w+$", meta_model.output_dir):
        raise InvalidArgumentError(
            "Output directory name can only contain alphanumeric characters and underscores.")


def __validate_required_args(meta_model) -> None:
    if not isinstance(meta_model.step, int) or meta_model.step <= 0:
        raise InvalidArgumentError("step should be a positive integer.")

    if not isinstance(meta_model.width, int) or meta_model.width <= meta_model.step:
        raise InvalidArgumentError(f"width should be greater than step = {meta_model.step}.")

    if meta_model.width_events is not None and not isinstance(meta_model.width_events,
                                                              (int, float)) and meta_model.width_events <= 0:
        raise InvalidArgumentError("width_events should be either a positive integer or positive float.")


def validate_args(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    validate_args_1(meta_model)
    validate_args_2(meta_model)
    validate_args_3(meta_model)
    validate_args_4(meta_model)
    validate_args_5(meta_model)

    if len(meta_model.hyperparams_mm_network) != 3:
        raise ValueError("hyperparams_mm_network must be a tuple of length 3")

    if not all(isinstance(val, int) for val in meta_model.hyperparams_mm_network[:-1]):
        raise ValueError("hyperparams_mm_network values must be integers except the last which is"
                         " the activation function (str)")

    if not isinstance(meta_model.save_models_as_dot_format, bool):
        raise InvalidArgumentError("Invalid save_models_as_dot_format parameter: must be a boolean.")

    if meta_model.dropout is None or not 0 <= meta_model.dropout < 1 or not isinstance(meta_model.dropout, float):
        raise InvalidArgumentError("Invalid dropout parameter: must be a float between 0 and 1.0.")


def validate_args_1(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    if meta_model.fill_nan not in [FILL_NAN_ZEROS, FILL_NAN_FFILL, FILL_NAN_BFILL, FILL_NAN_MEDIAN]:
        raise InvalidArgumentError(
            f"Invalid method for filling NaN values. Supported methods are"
            f"  {FILL_NAN_ZEROS}, {FILL_NAN_FFILL}, {FILL_NAN_BFILL}, and {FILL_NAN_MEDIAN}.")

    if not isinstance(meta_model.epochs, int) or meta_model.epochs <= 0:
        raise InvalidArgumentError("epochs should be a positive integer.")

    if not isinstance(meta_model.batch_size, int) or meta_model.batch_size <= 0:
        raise InvalidArgumentError("batch_size should be a positive integer.")

    if not isinstance(meta_model.t_max, float) and not isinstance(meta_model.t_max, int):
        raise InvalidArgumentError("t_max should be float/int.")

    if meta_model.t_max <= meta_model.w_s:
        raise InvalidArgumentError(f"t_max should be greater than w_s {meta_model.w_s}.")

    if not isinstance(meta_model.delta, (int, float)) or meta_model.delta <= 0:
        raise InvalidArgumentError("delta should be either a positive integer or positive float.")

    if not (0 < meta_model.s_h < 1):
        raise InvalidArgumentError("s_h should be a float between 0 and 1 exclusive.")

    if not isinstance(meta_model.epsilon, float) or not (0 < meta_model.epsilon <= 1):
        raise InvalidArgumentError("epsilon should be a positive number between 0 and 1.")

    if not isinstance(meta_model.pa, int) or meta_model.pa <= 0:
        raise InvalidArgumentError("pa should be a positive integer.")


def validate_args_2(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    if not isinstance(meta_model.t_r, float) or not (0 < meta_model.t_r <= 1):
        raise InvalidArgumentError("t_r should be a positive number between 0 and 1.")

    if meta_model.time_window is not None and (
            not isinstance(meta_model.time_window, (int, float)) or meta_model.time_window <= 0):
        raise InvalidArgumentError("time_window should be either a positive integer or positive float.")

    if not all(isinstance(model, (str, tuple)) and
               (isinstance(model, str) or (isinstance(model, tuple) and len(model) == 2 and isinstance(model[0],
                                                                                                       str) and
                                           isinstance(model[1], int)))
               for model in meta_model.models):
        raise InvalidArgumentError(
            "Invalid format for models. It should be a list of strings or tuples of (string, integer).")


def validate_model_type(model):
    if model not in [LSTM, GRU, CNN, RNN_BIDIRECTIONAL, CONV_LSTM1D, RNN_ENCODER_DECODER, CNN_RNN,
                     SELF_ATTENTION, FFN, TRANSFORMER]:
        raise InvalidArgumentError(
            f"Invalid model type {model}. Supported models are {LSTM}, {GRU}, {CNN}, {RNN_BIDIRECTIONAL},"
            f" {CONV_LSTM1D}, {RNN_ENCODER_DECODER}, {CNN_RNN}, {SELF_ATTENTION}, {TRANSFORMER}, and {FFN}.")


def validate_model_instances(model_instances):
    if not isinstance(model_instances, int) or model_instances <= 0:
        raise InvalidArgumentError("Number of model instances should be a positive integer.")


def validate_args_3(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    for model in meta_model.models:
        if isinstance(model, str):
            validate_model_type(model)
        elif isinstance(model, tuple) and len(model) == 2:
            model_type, model_instances = model
            validate_model_type(model_type)
            validate_model_instances(model_instances)
        else:
            raise InvalidArgumentError(f"Invalid model specification {model}.")


def validate_args_4(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    if meta_model.type_training not in [TYPE_TRAINING_AVERAGE, TYPE_TRAINING_FFN]:
        raise InvalidArgumentError(
            f"Invalid type of training technique. Supported techniques are "
            f"{TYPE_TRAINING_AVERAGE} and {TYPE_TRAINING_FFN}.")

    if meta_model.scaler not in [MIN_MAX_SCALER, STANDARD_SCALER, ROBUST_SCALER]:
        raise InvalidArgumentError(
            f"Invalid type of scaler technique. Supported techniques are {MIN_MAX_SCALER},"
            f" {STANDARD_SCALER} and {ROBUST_SCALER}.")

    if not isinstance(meta_model.use_kfold, bool):
        raise InvalidArgumentError("Invalid use_kfold parameter: must be a boolean.")

    if not 0 < meta_model.test_size < 1 or not isinstance(meta_model.test_size, float):
        raise InvalidArgumentError("Invalid test_size parameter: must be a float between 0 and 1.")

    if not 0 < meta_model.val_size < 1 or not isinstance(meta_model.val_size, float):
        raise InvalidArgumentError("Invalid val_size parameter: must be a float between 0 and 1.")

    if len(meta_model.hyperparams_transformer) != 5:
        raise ValueError("hyperparams_transformer must be a tuple of length 5")


def validate_args_5(meta_model) -> None:
    """
    Validate the arguments of the MetaModel.

    Args:
        meta_model (MetaModel): A MetaModel instance.

    Returns:
        None

    Raises:
        ValueError: If any of the arguments are invalid.
    """

    param1, param2, param3, param4, param5 = meta_model.hyperparams_transformer
    if not (all(isinstance(p, int) for p in [param1, param2, param3]) and isinstance(param4, bool) and isinstance(
            param5,
            str)):
        raise ValueError("hyperparams_transformer values must be Tuple[int, int, int, bool, str]")


def validate_ffn(meta_model) -> Tuple:
    hyperparams_ffn = meta_model.hyperparams_ffn
    l_ffn = len(hyperparams_ffn)
    print(l_ffn)

    if not 2 < l_ffn < 6:
        raise ValueError("hyperparams_ffn must be a tuple of length 3, 4 or 5")

    if l_ffn == 3:
        max_layers, min_neurons, max_neurons = hyperparams_ffn
        return 1, max_layers, min_neurons, max_neurons, "sigmoid"

    if l_ffn == 4:
        if isinstance(hyperparams_ffn[-1], str):
            max_layers, min_neurons, max_neurons, activation = hyperparams_ffn
            return 1, max_layers, min_neurons, max_neurons, activation
        else:
            min_layers, max_layers, min_neurons, max_neurons = hyperparams_ffn
            return min_layers, max_layers, min_neurons, max_neurons, "sigmoid"

    return hyperparams_ffn


def validate_cnn(meta_model) -> Tuple:
    hyperparams_cnn = meta_model.hyperparams_cnn
    l_cnn = len(hyperparams_cnn)

    if not 4 < l_cnn < 8:
        raise ValueError("hyperparams_cnn must be a tuple of length between 5 and 7.")

    if l_cnn == 5:
        min_f, max_f, min_k, max_k, max_layers = hyperparams_cnn
        return min_f, max_f, min_k, max_k, 1, max_layers, "relu"

    if l_cnn == 6:
        if isinstance(hyperparams_cnn[-1], str):
            min_f, max_f, min_k, max_k, max_layers, activation = hyperparams_cnn
            return min_f, max_f, min_k, max_k, 1, max_layers, activation
        else:
            min_f, max_f, min_k, max_k, min_layers, max_layers = hyperparams_cnn
            return min_f, max_f, min_k, max_k, min_layers, max_layers, "relu"

    return hyperparams_cnn


def validate_rnn(meta_model) -> Tuple:
    hyperparams_rnn = meta_model.hyperparams_rnn
    l_rnn = len(hyperparams_rnn)

    if not 2 < l_rnn < 6:
        raise ValueError("hyperparams_rnn must be a tuple of length 3, 4 or 5")

    if l_rnn == 3:
        max_layers, min_neurons, max_neurons = hyperparams_rnn
        return 1, max_layers, min_neurons, max_neurons, "tanh"

    if l_rnn == 4:
        if isinstance(hyperparams_rnn[-1], str):
            max_layers, min_neurons, max_neurons, activation = hyperparams_rnn
            return 1, max_layers, min_neurons, max_neurons, activation
        else:
            min_layers, max_layers, min_neurons, max_neurons = hyperparams_rnn
            return min_layers, max_layers, min_neurons, max_neurons, "tanh"

    return hyperparams_rnn

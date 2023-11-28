import os
import pprint
import shutil
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from eventdetector_ts import FFN, FILL_NAN_ZEROS, TYPE_TRAINING_AVERAGE, STANDARD_SCALER, \
    config_dict, CONFIG_FILE
from eventdetector_ts.data.helpers_data import compute_middle_event, remove_close_events, \
    convert_events_to_intervals, get_union_times_events, get_dataset_within_events_times, \
    convert_dataframe_to_overlapping_partitions, op, check_time_unit, save_dict_to_json, \
    convert_dataset_index_to_datetime, convert_seconds_to_time_unit
from eventdetector_ts.metamodel import logger_meta_model
from eventdetector_ts.metamodel.utils import DataSplitter, validate_args, validate_required_args, validate_ffn, \
    validate_cnn, validate_rnn
from eventdetector_ts.models.models_builder import ModelCreator
from eventdetector_ts.models.models_trainer import ModelTrainer
from eventdetector_ts.optimization.event_extraction_pipeline import OptimizationData, EventOptimization
from eventdetector_ts.plotter.plotter import Plotter


class MetaModel:
    def __init__(
            self,
            output_dir: str,
            dataset: pd.DataFrame,
            events: Union[list, pd.DataFrame],
            width: int,
            step: int = 1,
            width_events: Optional[Union[int, float]] = None,
            **kwargs
    ):
        """
        Initializes a new instance of the MetaModel class.

        Args:
            output_dir (str): The name or path of the directory where all outputs will be saved.
                If output_dir is a folder name, the full path in the current directory will be created.
            dataset (pd.DataFrame): The input dataset as a Pandas DataFrame.
            events (Union[list, pd.DataFrame]): The input events as either a list or a Pandas DataFrame.
            width (int): Number of consecutive time steps in each partition (window) when creating overlapping 
                partitions (sliding windows).
            step (int = 1): Number of time steps to advance the sliding window. Default to 1.
            width_events (Union[int, float] = None): The width of each event. 
                If it's an integer, it represents the number of time steps that constitute an event. 
                If it's a float, it represents the duration in seconds of each event. 
                If not provided (None), it defaults to the value of (width -1).
            kwargs (Dict): Optional keyword arguments for additional parameters.
                - t_max (float): The maximum total time is linked to the `sigma variable of the Gaussian filter. 
                    This time should be expressed in the same unit of time (seconds, minutes, etc.) as used in the 
                    dataset. The unit of time for the dataset is determined by its time sampling. In other words, 
                    the `sigma` variable should align with the timescale used in your time series data. 
                    The default value is calculated as (3 x (width-1) x time_sampling) / 2.
                - delta (Union[int, float]): The maximum time tolerance used to determine the correspondence 
                    between a predicted event and its actual counterpart in the true events. If it's an integer, it 
                    represents the number of time steps. If it's a float, it represents the duration in seconds
                    The default value is width_events x time_sampling.
                - s_h (float): A step parameter for adjusting the peak height threshold `h` during the peak detection 
                    process. The default value is 0.05.
                - epsilon (float): A small constant used to control the size of set which contains the top models
                    with the lowest MSE values. The default value is 0.0002.
                - pa (int): The patience for the early stopping algorithm. The default value is 5.
                - t_r (float): The ratio threshold for the early stopping algorithm.
                    The default value is 0.97.
                - time_window (Union[int, float] = None): This parameter controls the amount of data within the dataset 
                    is used for the training process. If it's an integer, it represents a specific number time steps. 
                    If it's a float, it represents a duration in seconds. By default, it is set to None, which means all
                    available data will be used. However, if a value is provided, the dataset will include a specific 
                    interval of data surrounding each reference event. This interval includes data from both sides of 
                    each event, with a duration equal to the specified `time_window`. Setting a `time_window` in some 
                    situations can offer several advantages, such as accelerating the training process and enhancing 
                    the neural networks' understanding of rare events.
                - models (List[Union[str, Tuple[str, int]]]): Determines the type of deep learning models to use.
                    If a tuple is passed, it specifies both the model type and the number of instances to run.
                    The default value is [(FFN, 2)].
                - hyperparams_ffn (Tuple[int, int, int, int, str]): Specify for the FFN the minimum and the maximum 
                    number of layers, the minimum and the maximum number of neurons per layer, and the activation 
                    function. The default value is (1, 3, 64, 256, "sigmoid"). The List of available activation 
                    functions are ["relu","sigmoid","tanh","softmax","leaky_relu","elu","selu","swish"]. 
                    If you pass `None`, no activation is applied (i.e. "linear" activation: `a(x) = x`).
                - hyperparams_cnn (Tuple[int, int, int, int, int, str]): Specify for the CNN the minimum, maximum number
                    of filters, the minimum, the maximum kernel size, the minimum and the maximum number of pooling
                    layers, and the activation function. The default value is (16, 64, 3, 8, 1, 2, "relu").
                - hyperparams_transformer (Tuple[int, int, int, bool, str]): Specify for Transformer the Key dimension, 
                    number of heads, the number of the encoder blocks, a flag to indicate the use of the original 
                    architecture, and the activation function. The default value is (256, 8, 10, True, "relu").
                - hyperparams_rnn (Tuple[int, int, int, str]): Specify for the RNN the minimum and the maximum number 
                    of RNN layers, the minimum and the maximum number of hidden units, and the activation function.
                    The default value is (1, 2, 16, 128, "tanh").
                - hyperparams_mm_network (Tuple[int, int, str]): Specify for the MetaModel network the number
                    of layers, the number of neurons per layer, and the activation function.
                    The default value is (1, 32, "sigmoid").
                - epochs (int): The number of epochs to train different models. The default value is False 256.
                - batch_size (int): The number of samples per gradient update.
                    The default value is 32.
                - fill_nan (str): Specifies the method to use for filling NaN values in the dataset.
                    Supported methods are 'zeros', 'ffill', 'bfill', and 'median'.
                    The default is 'zeros'.
                - type_training (str):Specifies the type of training technique to use for the MetaModel.
                    Supported techniques are 'average' and 'ffn'.
                    The default is 'average'.
                - scaler (str): The type of scaler to use for preprocessing the data.
                    Possible values are "MinMaxScaler", "StandardScaler", and "RobustScaler".
                    Default is "StandardScaler"
                - use_kfold (bool): Whether to use k-fold cross-validation technique or not.
                The default value is False.
                - test_size (float): The proportion of the dataset to include in the test split.
                    Should be a value between 0 and 1. Default is 0.2.
                - val_size (float): The proportion of the training set to use for validation.
                    Should be a value between 0 and 1. Default is 0.2.
                - save_models_as_dot_format (bool = False): Whether to save the models as a dot format file.
                    The default value is False. If set to True, then you should have graphviz software
                    to be installed on your machine.
                - remove_overlapping_events (bool = True): Whether to remove the overlapping events or not. 
                    The default value is True.
                - dropout (float = 0.3): The dropout rate, which determines the fraction of input units to drop during 
                    training.
                - last_act_func (str = "sigmoid"): Activation function for the final layer of each model. Defaults to
                    "sigmoid". If set to `None`, no activation will be applied (i.e., "linear" activation: `a(x) = x`).

        """
        self.step = step
        self.width = width
        self.events = events
        self.dataset = dataset
        self.output_dir = output_dir
        self.width_events = width_events
        validate_required_args(self)
        self.kwargs: Dict = kwargs
        self.y = np.empty(shape=(0,))
        self.x = np.empty(shape=(0,))
        self.__compute_and_set_time_sampling()
        self.__set_defaults()
        validate_args(self)

        if self.save_models_as_dot_format:
            logger_meta_model.warning("save_models_as_dot_format is set to true, "
                                      "you should have graphviz software to be installed on your machine.")
        self.__create_output_dir()
        # Create a `ModelCreator` object with the provided models and hyperparameters
        self.model_creator: ModelCreator = ModelCreator(models=self.models, hyperparams_ffn=self.hyperparams_ffn,
                                                        hyperparams_cnn=self.hyperparams_cnn,
                                                        hyperparams_rnn=self.hyperparams_rnn,
                                                        hyperparams_transformer=self.hyperparams_transformer,
                                                        last_act_func=self.last_act_func, dropout=self.dropout,
                                                        save_models_as_dot_format=self.save_models_as_dot_format,
                                                        root_dir=self.output_dir)
        # Create a `DataSplitter` object with the provided test_size and scaler_type
        self.data_splitter: DataSplitter = DataSplitter(test_size=self.test_size, scaler_type=self.scaler)
        # Create a `ModelTrainer` object with the provided data_splitter, epochs,
        #   batch_size, pa, t_r, use_kfold, val_size, epsilon and save_models_as_dot_format.
        self.model_trainer: ModelTrainer = ModelTrainer(data_splitter=self.data_splitter, epochs=self.epochs,
                                                        batch_size=self.batch_size, pa=self.pa, t_r=self.t_r,
                                                        use_kfold=self.use_kfold,
                                                        val_size=self.val_size, epsilon=self.epsilon,
                                                        save_models_as_dot_format=self.save_models_as_dot_format)
        # class represents the data used for the event extraction pipeline.
        self.optimization_data: OptimizationData = OptimizationData(t_max=self.t_max, w_s=self.w_s, s_s=self.s_s,
                                                                    s_h=self.s_h, delta=self.delta,
                                                                    output_dir=self.output_dir,
                                                                    time_unit=self.time_unit)

        self.event_optimization: EventOptimization = EventOptimization(optimization_data=self.optimization_data)
        # The Plotter class is responsible for generating and saving plots.
        self.plotter: Plotter = Plotter(root_dir=self.output_dir, time_unit=self.time_unit,
                                        width_events_s=self.width_events_s)

    def __create_output_dir(self) -> None:
        """
           Check if output_dir is already a complete path, if output_dir is a folder name,
            create the full path in the current directory.

           Returns:
               None
           """

        # Check if output_dir is already a complete path
        if os.path.isabs(self.output_dir):
            if not os.path.exists(self.output_dir):
                logger_meta_model.critical(f"{self.output_dir} does not exists")
                raise ValueError(f"{self.output_dir} does not exists")

        # If output_dir is a folder name, create the full path in the current directory
        else:
            # Get the absolute path of the current directory
            current_directory = os.path.abspath(".")
            self.output_dir = os.path.join(current_directory, self.output_dir)
            if os.path.exists(self.output_dir):
                logger_meta_model.warning(f"The working directory '{self.output_dir}' exists and it will be deleted")
                shutil.rmtree(self.output_dir)
            logger_meta_model.info(f"Creating the working directory at: '{self.output_dir}'")
            os.makedirs(self.output_dir)

        config_dict['output_dir'] = self.output_dir

    def __set_defaults_bis(self) -> None:
        """
        Sets default values for any missing keyword arguments in self.kwargs.

        Returns:
            None
        """
        if self.width_events is None:
            self.width_events = self.width
        self.t_max = self.kwargs.get('t_max', (3.0 * self.w_s) / 2)  # the minimum should be equal to w_s

        if self.kwargs.get('delta') is None:
            self.delta = self.width_events_s
        else:
            if isinstance(self.kwargs.get('delta'), float):
                self.delta = convert_seconds_to_time_unit(value=self.kwargs.get('delta'), unit=self.time_unit)
            else:
                self.delta = self.kwargs.get('delta') * self.t_s

        self.s_h = self.kwargs.get('s_h', 0.05)
        self.epsilon = self.kwargs.get('epsilon', 0.0002)
        self.pa = self.kwargs.get('pa', 5)
        self.t_r = self.kwargs.get('t_r', 0.97)

    def __set_defaults(self) -> None:
        """
        Sets default values for any missing keyword arguments in self.kwargs.

        Returns:
            None
        """
        self.__set_defaults_bis()

        if self.kwargs.get('time_window') is None:
            self.time_window = None
        else:
            if isinstance(self.kwargs.get('time_window'), float):
                self.time_window = convert_seconds_to_time_unit(value=self.kwargs.get('time_window'),
                                                                unit=self.time_unit)
            else:
                self.time_window = self.kwargs.get('time_window') * self.t_s

        self.models = self.kwargs.get('models', [(FFN, 2)])
        for i, model in enumerate(self.models):
            if isinstance(model, str):
                self.models[i] = (model, 1)
            elif isinstance(model, tuple) and len(model) == 1:
                self.models[i] = (model[0], 1)

        self.hyperparams_ffn = self.kwargs.get('hyperparams_ffn', (1, 3, 64, 256, "sigmoid"))
        self.hyperparams_ffn = validate_ffn(self)
        self.hyperparams_cnn = self.kwargs.get('hyperparams_cnn', (16, 64, 3, 8, 1, 2, "relu"))
        self.hyperparams_cnn = validate_cnn(self)
        self.hyperparams_rnn = self.kwargs.get('hyperparams_rnn', (1, 2, 16, 128, "tanh"))
        self.hyperparams_rnn = validate_rnn(self)
        self.hyperparams_transformer = self.kwargs.get("hyperparams_transformer", (256, 4, 1, True, "relu"))
        self.hyperparams_mm_network = self.kwargs.get('hyperparams_mm_network', (1, 32, "sigmoid"))
        self.epochs = self.kwargs.get('epochs', 256)
        self.batch_size = self.kwargs.get('batch_size', 32)
        self.fill_nan = self.kwargs.get('fill_nan', FILL_NAN_ZEROS)
        self.type_training = self.kwargs.get('type_training', TYPE_TRAINING_AVERAGE)
        self.scaler = self.kwargs.get('scaler', STANDARD_SCALER)
        self.use_kfold = self.kwargs.get('use_kfold', False)
        self.test_size = self.kwargs.get('test_size', 0.2)
        self.val_size = self.kwargs.get('val_size', 0.2)

        self.save_models_as_dot_format = self.kwargs.get('save_models_as_dot_format', False)
        self.remove_overlapping_events = self.kwargs.get("remove_overlapping_events", True)
        self.last_act_func = self.kwargs.get("last_act_func", "sigmoid")
        self.dropout = self.kwargs.get("dropout", 0.3)

        log_dict = {
            'width_events_s': self.width_events_s,
            't_max': self.t_max,
            'delta': self.delta,
            's_h': self.s_h,
            'epsilon': self.epsilon,
            'pa': self.pa,
            't_r': self.t_r,
            'time_window': self.time_window,
            'models': self.models,
            'hyperparams_ffn': self.hyperparams_ffn,
            'hyperparams_cnn': self.hyperparams_cnn,
            'hyperparams_rnn': self.hyperparams_rnn,
            'hyperparams_transformer': self.hyperparams_transformer,
            'hyperparams_mm_network': self.hyperparams_mm_network,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'fill_nan': self.fill_nan,
            'type_training': self.type_training,
            'scaler': self.scaler,
            'use_kfold': self.use_kfold,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'save_models_as_dot_format': self.save_models_as_dot_format,
            "remove_overlapping_events": self.remove_overlapping_events,
            "last_act_func": self.last_act_func,
            "dropout": self.dropout
        }

        log_message = pprint.pformat(log_dict, indent=4)
        logger_meta_model.info(log_message)

        config_dict.update({'width': self.width, 'step': self.step, 'batch_size': self.batch_size,
                            'type_training': self.type_training, 'fill_nan': self.fill_nan})

    def __compute_and_set_time_sampling(self) -> None:
        """
        Compute the time sampling of the dataset by calculating the time difference between the first two index values.
                Then set the corresponding parameters: t_s, w_s, and s_s.

        Returns:
            None

        Raises:
            TypeError: If the index of the dataset is not in datetime format.
        """
        try:
            logger_meta_model.info("checks if the index of the dataset is already in the datetime format.")
            convert_dataset_index_to_datetime(self.dataset)
            # Get the first two index values of the dataset
            a = self.dataset.index[0]
            b = self.dataset.index[1]
            # Calculate the time difference between the first two index values
            diff = b - a
            # Check the units of the time difference
            logger_meta_model.info("Computing the time sampling and time unit of the dataset")
            self.t_s, self.time_unit = check_time_unit(diff=diff)
            logger_meta_model.warning(f"The time sampling t_s is {self.t_s} {self.time_unit}s")
            self.w_s = self.t_s * (self.width - 1)
            self.s_s = self.t_s * self.step

            if self.width_events is None:
                self.width_events_s = self.w_s
            else:
                self.width_events_s = self.t_s * self.width_events

            if isinstance(self.width_events, float):
                self.width_events_s = convert_seconds_to_time_unit(value=self.width_events, unit=self.time_unit)

            config_dict['w_s'] = self.w_s
            config_dict['width_events_s'] = self.width_events_s
            config_dict['time_unit'] = self.time_unit.value
        except AttributeError:
            logger_meta_model.critical("The dataset is not compatible with the datetime format")
            raise TypeError("The index should be in datetime format.")

    def prepare_data_and_computing_op(self) -> None:
        """
        Prepare the events and dataset for computing op.
        This method will compute the middle event of the given events, remove any close events based on the self.w_s,
            and convert the remaining events to intervals. If a time partition is specified, it will get the union of
            event times and extract the corresponding portion of the dataset.

        The dataset will then be converted to overlapping partitions using the specified width and step size, 
        and the $op$ (overlapping parameter) values will be computed for each partition based on the given intervals.

        Finally, the learning data (overlapping partitions and corresponding $op$ values) will be stored in
            the instance variables x and y.

        Returns:
             None
        """

        logger_meta_model.info("Computes the middle date of events...")

        self.events = compute_middle_event(self.events)

        logger_meta_model.info("Removes events that occur too close together...")
        temp: int = len(self.events)
        self.events = remove_close_events(self.events, self.width_events_s, self.time_unit,
                                          self.remove_overlapping_events)

        logger_meta_model.warning(f"A total of {temp - len(self.events)}/{temp} events were removed due to overlapping")
        logger_meta_model.info("Convert events to intervals...")
        intervals = convert_events_to_intervals(self.events, self.width_events_s, self.time_unit)

        if self.time_window is not None:
            logger_meta_model.warning(f"time_window is provided = {self.time_window} {self.time_unit}s")
            events_times = get_union_times_events(self.events, self.time_window, self.time_unit)
            self.dataset = get_dataset_within_events_times(self.dataset, events_times)

        logger_meta_model.info("Computing overlapping partitions...")
        overlapping_partitions = convert_dataframe_to_overlapping_partitions(self.dataset, width=self.width,
                                                                             step=self.step,
                                                                             fill_method=self.fill_nan)

        logger_meta_model.info("Computing op...")
        self.x, self.y = op(dataset_as_overlapping_partitions=overlapping_partitions, events_as_intervals=intervals)

        # Convert x and y arrays to float32 for consistency
        self.x = np.asarray(self.x).astype('float32')
        self.y = np.asarray(self.y).astype('float32')

        self.optimization_data.set_overlapping_partitions(overlapping_partitions)
        self.optimization_data.set_true_events(self.events)

    def build_stacking_learning(self) -> None:
        """
        Builds a stacking learning pipeline using the provided models and hyperparameters.

        Returns:
            None
        """

        # Get the number of time steps and features from the x data
        n_time_steps, n_features = self.x.shape[1], self.x.shape[2]
        config_dict['n_time_steps'] = n_time_steps
        inputs = tf.keras.Input(shape=(n_time_steps, n_features), name="input")
        # Call the `create_models` method to create the models
        logger_meta_model.info(f"Create the following models: {list(map(lambda x: x[0], self.models))}")
        self.model_creator.create_models(inputs=inputs)
        logger_meta_model.info("Split the data into training, validation, and test sets and apply "
                               "the specified scaler to each time step...")
        self.data_splitter.split_data_and_apply_scaler(x=self.x, y=self.y)
        logger_meta_model.info("Saves the scalers to disk...")
        self.data_splitter.save_scalers(output_dir=self.output_dir)
        logger_meta_model.info("Fits the created models to the training data...")
        self.model_trainer.fitting_models(self.model_creator.created_models)
        logger_meta_model.info("Saving the best models...")
        self.model_trainer.save_best_models(output_dir=self.output_dir)
        predicted_y, loss, test_y = self.model_trainer.train_meta_model(type_training=self.type_training,
                                                                        hyperparams_mm_network
                                                                        =self.hyperparams_mm_network,
                                                                        output_dir=self.output_dir)
        self.optimization_data.set_predicted_op(predicted_op=predicted_y)
        logger_meta_model.info(f"The loss of the MetaModel is {loss:.4f}")
        self.plotter.set_data_op(test_y=test_y, predicted_y=predicted_y)
        self.plotter.set_losses(train_losses=self.model_trainer.train_losses,
                                val_losses=self.model_trainer.val_losses, train_loss_meta_model=
                                self.model_trainer.train_loss_meta_model,
                                val_loss_meta_model=self.model_trainer.val_loss_meta_model)

    def event_extraction_optimization(self) -> None:
        """
        Run the Event Extraction Optimization process.

        Returns:
            None
        """

        predicted_events, delta_t = self.event_optimization.max_f1score()
        path = os.path.join(self.output_dir, CONFIG_FILE)
        logger_meta_model.info(f"Saving config file into {path}")
        save_dict_to_json(path=path, data=config_dict)
        self.plotter.set_data_events(predicted_events=predicted_events, true_events=self.optimization_data.true_events)
        self.plotter.set_delta_t(delta_t=delta_t)

    def plot_save(self, show_plots: bool = True) -> None:
        """
        Plot the results: losses, true/predicted op, true/predicted events, deltat_t.

        Args:
            show_plots (bool): whether to show the plots or not.
            
        Returns:
            None
        """
        self.plotter.set_show(show=show_plots)
        self.plotter.plot_losses()
        self.plotter.plot_prediction()
        self.plotter.plot_predicted_events()
        self.plotter.plot_delta_t(bins=10)

    def fit(self) -> None:
        """
        Run prepare_data_and_computing_op, build_stacking_learning, event_extraction_optimization, and plot_save
        
        Returns:
            None
        """
        self.prepare_data_and_computing_op()
        self.build_stacking_learning()
        self.event_extraction_optimization()
        self.plot_save()

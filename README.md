Universal Approximation for Event Detection in Time Series
==========================================================

Welcome to **Event Detector**, a Python package for detecting events in time series data. The emphasis of this package
is on offering useful machine learning functionalities, such as customizing and fitting the model on multidimensional
time series, training on large datasets, ensemble models, and providing rich support for event detection in time
series.

To get started using **Event Detector**, simply follow the instructions below to install the required packages and
dependencies.

1. Clone the repository:

<pre><code>git clone https://github.com/menouarazib/eventdetector.git
cd eventdetector
</code></pre>

2. Create a virtual environment:

<pre><code>python -m venv env
source env/bin/activate  # for Linux/MacOS
env\Scripts\activate.bat  # for Windows
</code></pre>

3. Install the required packages:

<pre><code>pip install -r requirements.txt</code></pre>

4. Quickstart Example

To quickly get started with the Event Detection in Time Series package, follow the steps below:

- Download the dataset and events using the links provided:
    - [mex_dataset_2012](https://drive.google.com/file/d/1v8W50aveNMUeofDOQoI_601E0IN990BS/view?usp=sharing)
    - [mex_events](https://drive.google.com/file/d/1cMZn4fsgot2J2EffNCKvm0I2XKiIemkl/view?usp=sharing)

```python
from datetime import datetime

import pandas as pd

from eventdetector.metamodel.meta_model import MetaModel
from eventdetector.prediction.prediction import predict
from eventdetector.prediction.utils import plot_prediction

# Get the dataset.
dataset_mex: pd.DataFrame = pd.read_pickle("mex_dataset_2012.pkl")

# Get events.
events: pd.DataFrame = pd.read_pickle("mex_events.pkl")

# Select dates for learning phase.
start_date = datetime(2012, 1, 1)
stop_date = datetime(2012, 10, 1)

# Filtering dataset using learning dates.
dataset_mex_learning = dataset_mex[(dataset_mex.index >= start_date) & (dataset_mex.index <= stop_date)]
print(dataset_mex_learning)

# Filtering events by using learning dates.
events_learning = events[
    (events['event'] >= start_date) & (events['event'] <= stop_date)]

"""
The 'time_window' parameter is crucial for controlling the amount of data used in the dataset. It should be specified 
as a number of units of time. By default, it is set to None, which means that all available data will be used.
However, if a value is provided, the dataset will only include a specific interval of data around each reference event.
This interval consists of data from both the left and right sides of each event, with a duration equal to the specified 
time_window. Setting a time_window can offer several advantages, including speeding up the training process and 
improving the neural networks' understanding for rare events.
"""
time_window: int = 5400

# Create the MetaModel.
meta_model = MetaModel(output_dir="mex_bow_shocks", dataset=dataset_mex_learning, events=events_learning,
                       width=45, step=1, time_window=time_window, batch_size=3000)

# Prepare the events and dataset for computing op.
meta_model.prepare_data_and_computing_op()

# Builds a stacking learning pipeline using the provided models and hyperparameters.
meta_model.build_stacking_learning()

# Run the Event Extraction Optimization process.
meta_model.event_extraction_optimization()

# Plot the results: Losses, true/predicted op, true/predicted events, deltat_t.
meta_model.plot_save(show_plots=True)

# Make Predictions:

# Select dates for predictions.
start_date_prediction = datetime(2012, 10, 1)
stop_date_prediction = datetime(2013, 1, 1)

# Filtering dataset using prediction dates.
dataset_mex_prediction = dataset_mex[
    (dataset_mex.index >= start_date_prediction) & (dataset_mex.index <= stop_date_prediction)]
print(dataset_mex_prediction)

# Provide the absolute path for the folder created by the MetaModel.
path: str = '/home/.../mex_bow_shocks/'

# Call the 'predict' method
predicted_events, predicted_op, filtered_predicted_op = predict(dataset=dataset_mex_prediction, path=path)

# Plot the predicted Op
plot_prediction(predicted_op=predicted_op, filtered_predicted_op=filtered_predicted_op)
```

5. MetaModel Arguments:

Argument | Description
   ---------------------------- | --------------------------------------------------------------
   output_dir | The name or path of the directory where all outputs will be saved. If `output_dir` is a folder name, it will create the full path.
   dataset | The input dataset as `pd.DataFrame`.
   events | The input events as a list or `pd.DataFrame`.
   width | The width to be used for creating sliding windows.
   step | The step size between two successive windows.
   kwargs | Optional keyword arguments:
   t_max | The maximum total time related to sigma. Default: `(3 * w_s) / 2)`.
   delta | The maximum time tolerance used to determine the correspondence between a predicted event and its actual counterpart. Default: `w_s`.
   s_h | A step parameter for the peak height threshold `h`. Default: `0.05`.
   epsilon | A small constant used to control the size of the set containing top models with the lowest MSE values. Default: `0.0002`.
   pa | The patience for the early stopping algorithm. Default: `5`.
   t_r | The ratio threshold for the early stopping algorithm. Default: `0.97`.
   time_window | The 'time_window' parameter is crucial for controlling the amount of data used in the dataset. It should be specified as a number of units of time. By default, it is set to `None`, which means that all available data will be used. However, if a value is provided, the dataset will only include a specific interval of data around each reference event. This interval consists of data from both the left and right sides of each event, with a duration equal to the specified `time_window`. Setting a time_window can offer several advantages, including speeding up the training process and improving the neural networks' understanding for rare events.
   models | Determine the type of deep learning models and the number of instances to use. Default: `[(model, 2) for model in [FFN]]`.
   hyperparams_ffn | Specify for the FFN the maximum number of layers, the minimum and the maximum number of neurons per layer. Default: `(3, 64, 256)`.
   hyperparams_cnn | Specify for the CNN the minimum and maximum number of filters, the minimum, the maximum kernel size, and maximum number of pooling layers. Default: `(16, 64, 3, 8 , 2)`.
   hyperparams_rnn | Specify for the RNN the maximum number of RNN layers, the minimum and the maximum number of hidden units. Default: `(1, 16, 128)`.
   hyperparams_mm_network | Specify for the MetaModel network the number of layers and the number of neurons per layer. Default: `(1, 32)`.
   epochs | The number of epochs to train different models. Default: `256`.
   batch_size | The number of samples per gradient update. Default: `32`.
   fill_nan | Specifies the method to use for filling `NaN` values in the dataset. Supported methods are 'zeros', 'ffill', 'bfill', and 'median'. Default: `"zeros"`.
   type_training | Specify the type of training technique to use for the MetaModel. Supported techniques are 'average' and 'ffn'. Default: `"average"`.
   scaler | The type of scaler to use for preprocessing the data. Possible values are "MinMaxScaler", "StandardScaler", and "RobustScaler". Default: `"StandardScaler"`.
   use_kfold | Whether to use k-fold cross-validation technique or not. Default: `False`.
   test_size | The proportion of the dataset to include in the test split. Should be between 0 and 1. Default is `0.2`.
   val_size |The proportion of the training set to use for validation. Should be a value between 0 and 1. Default is `0.2`.
   use_multiprocessing | Whether to use multiprocessing or not for the event exctraction optimization. The default value is `False`.
   save_models_as_dot_format | Whether to save the models as a dot format file. The default value is `False`. If set to True, then you should have `graphviz` software to be installed on your machine.

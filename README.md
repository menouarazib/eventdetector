<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/eventdetector/a4d7e137f88a4a476ba6d07f43337ec39543a522/images/logo_eventdetector.svg" width="400">
</h1><br>

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/eventdetector-ts.svg?color=brightgreen)](https://pypi.org/project/eventdetector-ts/)
![Unit Tests and Lint](https://github.com/menouarazib/eventdetector/actions/workflows/unit_tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/menouarazib/eventdetector/badge.svg?branch=master)](https://coveralls.io/github/menouarazib/eventdetector?branch=master)
[![License](https://img.shields.io/github/license/menouarazib/eventdetector)](https://github.com/menouarazib/eventdetector/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.31219/osf.io/uabjg.svg)](https://doi.org/10.31219/osf.io/uabjg)

Universal Event Detection in Time Series
==========================================================
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Make Prediction](#make-prediction)
- [Documentation](#documentation)
- [How to credit our package](#how-to-credit-our-package)

## Introduction
Welcome to **EventDetector**, a Python package for detecting events in time series data. The emphasis of this package
is on offering useful machine learning functionalities, such as customizing and fitting the model on multidimensional
time series, training on large datasets, ensemble models, and providing rich support for event detection in time
series.

## Installation
### PyPi installation
<pre><code>
pip install eventdetector-ts</code>
</pre>
### Manual installation
To get started using **Event Detector**, simply follow the instructions below to install the required packages and
dependencies.
#### Clone the repository:

<pre><code>git clone https://github.com/menouarazib/eventdetector.git
cd eventdetector
</code></pre>

#### Create a virtual environment:

<pre><code>python -m venv env
source env/bin/activate  # for Linux/MacOS
env\Scripts\activate.bat  # for Windows
</code></pre>

#### Install the required packages:

<pre><code>pip install -r requirements.txt</code></pre>

## Quickstart Example
### Data sets
To quickly get started with the Event Detection in Time Series package, follow the steps below:

- You can either download the datasets and event catalogs manually or use the built-in methods for the desired application:
  - Martian Bow Shock: `eventdetector_ts.load_martian_bow_shock()`
      - [bow_shock_dataset](https://archive.org/download/martian_bow_shock_dataset/martian_bow_shock_dataset.pkl)
      - [bow_shock_events](https://archive.org/download/martian_bow_shock_events/martian_bow_shock_events.csv)
  - Credit Card Fraud: `eventdetector_ts.load_credit_card_fraud()`
      - [credit_card_dataset](https://archive.org/download/credit_card_fraud_dataset/credit_card_fraud_dataset.csv)
      - [credit_card_events](https://archive.org/download/credit_card_fraud_events/credit_card_fraud_events.csv)
### Code Implementations:
  - Credit Card Fraud:
    ```python
    from eventdetector_ts import load_credit_card_fraud
    from eventdetector_ts.metamodel.meta_model import MetaModel
    
    dataset, events = load_credit_card_fraud()
    meta_model = MetaModel(dataset=dataset, events=events, width=3, step=1, output_dir='fraud', batch_size=3000)
    # Prepare the events and dataset for computing op.
    meta_model.prepare_data_and_computing_op()
    # Builds a stacking learning pipeline using the provided models and hyperparameters.
    meta_model.build_stacking_learning()
    # Run the Event Extraction Optimization process.
    meta_model.event_extraction_optimization()
    # Plot the results: Losses, true/predicted op, true/predicted events, deltat_t.
    meta_model.plot_save(show_plots=True)
    ```

  - Martian Bow Shock:
    ```python
    from eventdetector_ts import load_martian_bow_shock
    from eventdetector_ts.metamodel.meta_model import MetaModel
    
    dataset, events = load_martian_bow_shock()
    
    # Create the MetaModel.
    meta_model = MetaModel(output_dir="martian_bow_shocks", dataset=dataset, events=events, width=45, step=1,
                           time_window=5400, batch_size=3000)
    # Prepare the events and dataset for computing op.
    meta_model.prepare_data_and_computing_op()
    # Builds a stacking learning pipeline using the provided models and hyperparameters.
    meta_model.build_stacking_learning()
    # Run the Event Extraction Optimization process.
    meta_model.event_extraction_optimization()
    # Plot the results: Losses, true/predicted op, true/predicted events, deltat_t.
    meta_model.plot_save(show_plots=True)
    ```
### Results and Performance Evaluation

#### Performance Metrics

Table below presents the performance metrics for precision, recall, and F1-Score, providing a quantitative assessment of the framework's accuracy and effectiveness in the two data sets.

| Data set          | F1-Score | Precision | Recall   |
|-------------------|----------|-----------|----------|
| Martian bow shock | 0.9021   | 0.9455    | 0.8626   |
| Credit card fraud | 0.8372   | 0.9643    | 0.7397   |

#### Training and Validation Losses

The Figure below showcases the training loss and validation loss of the stacked models during the training process on the Martian bow shock and credit card fraud cases. The stacked models used in this evaluation consist of two feedforward neural networks (`FFN_0`, `FFN_1`) with distinct configurations of hyperparameters. The low losses observed in both cases indicate that the meta model has successfully learned the underlying patterns, justifying the obtained good metrics.

![Training and validation losses](https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/losses_mex_ccf.png)

#### Comparison of Predicted `op` and True `op`
The Figure below illustrates the comparison between the predicted $op$ values and the true $op$ values on the Martian bow shock (`delta = 180` seconds) and credit card fraud (`delta = 3` seconds) datasets.

![Comparison of predicted `op` and true `op`](https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/op_mex_ccf.png)

## Make Prediction
```python
from eventdetector_ts.prediction.prediction import predict
from eventdetector_ts.prediction.utils import plot_prediction

dataset_for_prediction = ...

# Call the 'predict' method
predicted_events, predicted_op, filtered_predicted_op = predict(dataset=dataset_for_prediction,
                                                                path='path to output_dir')
# Plot the predictions
plot_prediction(predicted_op=predicted_op, filtered_predicted_op=filtered_predicted_op)
```
 

## Documentation
### Meta Model
The first thing to do is to instantiate the `MetaModel` object with the required args:
```python
from eventdetector_ts.metamodel.meta_model import MetaModel

meta_model = MetaModel(output_dir=..., dataset=..., events=..., width=..., step=...)
```
For a complete description of the required and optional arguments, please refer to the following tables:

#### Required Arguments
| Argument       | Type                      | Description                                                                                                                                                                                                                                                  | Default Value |
|----------------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `output_dir`   | str                       | The name or path of the directory where all outputs will be saved. If `output_dir` is a folder name, the full path in the current directory will be created.                                                                                                 | -             |
| `dataset`      | pd.DataFrame              | The input dataset as a Pandas DataFrame.                                                                                                                                                                                                                     | -             |
| `events`       | Union[list, pd.DataFrame] | The input events as either a list or a Pandas DataFrame.                                                                                                                                                                                                     | -             |
| `width`        | int                       | Number of consecutive time steps in each partition (window) when creating overlapping partitions (sliding windows).                                                                                                                                          | -             |
| `step`         | int                       | Number of time steps to advance the sliding window.                                                                                                                                                                                                          | 1             |
| `width_events` | Union[int, float]         | The width of each event. If it's an `ìnt`, it represents the number of time steps that constitute an event. If it's a `float`, it represents the duration in seconds of each event. If not provided (None), it defaults to the value of width*time_sampling. | `width`       |

#### Optional Arguments: Kwargs
| Argument                    | Type                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Default Value                     |
|-----------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `t_max`                     | float                               | The maximum total time is linked to the `sigma` variable of the Gaussian filter. This time should be expressed in the same unit of time (seconds, minutes, etc.) as used in the dataset. The unit of time for the dataset is determined by its time sampling. In other words, the `sigma` variable should align with the timescale used in your time series data.                                                                                                                                                                                                                                                                                                                                                               | (3 * `width` x time_sampling) / 2 |
| `delta`                     | Union[int, float]                   | The maximum time tolerance used to determine the correspondence between a predicted event and its actual counterpart in the true events. If it's an integer, it represents the number of time steps. If it's a float, it represents the duration in seconds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `width_events` * time_sampling    |
| `s_h`                       | float                               | A step parameter for adjusting the peak height threshold `h` during the peak detection process.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 0.05                              |
| `epsilon`                   | float                               | A small constant used to control the size of set which contains the top models with the lowest MSE values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 0.0002                            |
| `pa`                        | int                                 | The patience for the early stopping algorithm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 5                                 |
| `t_r`                       | float                               | The ratio threshold for the early stopping algorithm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 0.97                              |
| `time_window`               | Union[int, float]                   | This parameter controls the amount of data within the dataset is used for the training process. If it's an integer, it represents a specific number time steps.  If it's a float, it represents a duration in seconds. By default, it is set to None, which means all available data will be used. However, if a value is provided, the dataset will include a specific interval of data surrounding each reference event. This interval includes data from both sides of each event, with a duration equal to the specified `time_window`. Setting a `time_window` in some situations can offer several advantages, such as accelerating the training process and enhancing the neural networks' understanding of rare events. | None                              |
| `models`                    | List[Union[str, Tuple[str, int]]]   | Determines the type of deep learning models and the number of instances to use. Available models: `LSTM`, `GRU`, `CNN`, `RNN_BIDIRECTIONAL`, `RNN_ENCODER_DECODER`, `CNN_RNN`, `FFN`, `CONV_LSTM1D`, `SELF_ATTENTION`, `TRANSFORMER`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `[(FFN, 2)]`                      |
| `hyperparams_ffn`           | Tuple[int, int, int, str]           | Specify for the FFN the maximum number of layers, the minimum and the maximum number of neurons per layer, and the activation function. The List of available activation functions are ["relu","sigmoid","tanh","softmax","leaky_relu","elu","selu","swish"]. If you pass `None`, no activation is applied (i.e. "linear" activation: `a(x) = x`).                                                                                                                                                                                                                                                                                                                                                                              | (3, 64, 256, "sigmoid")           |
| `hyperparams_cnn`           | Tuple[int, int, int, int, int, str] | Specify for the CNN the minimum and maximum number of filters, the minimum, the maximum kernel size, the maximum number of pooling layers, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (16, 64, 3, 8 , 2, "relu")        |
| `hyperparams_transformer`   | Tuple[int, int, int, bool, str]     | Specify for Transformer the Key dimension, number of heads, the number of the encoder blocks, a flag to indicate the use of the original architecture, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | (256, 8, 10, True, "relu")        |
| `hyperparams_rnn`           | Tuple[int, int, int, str]           | Specify for the RNN the maximum number of RNN layers,the minimum and the maximum number of hidden units, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | (1, 16, 128,"tanh")               |
| `hyperparams_mm_network`    | Tuple[int,int,str]                  | Specify for the MetaModel network the number of layers,the number of neurons per layer, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | (1 ,32,"sigmoid")                 |
| `epochs`                    | int                                 | The number of epochs to train different models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 256                               |
| `batch_size`                | int                                 | The number of samples per gradient update.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 32                                |
| `fill_nan`                  | str                                 | Specifies the method to use for filling `NaN` values in the dataset. Supported methods are 'zeros', 'ffill', 'bfill', and 'median'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | "zeros"                           |
| `type_training`             | str                                 | Specifies the type of training technique to use for the MetaModel. Supported techniques are 'average' and 'ffn'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | "average"                         |
| `scaler`                    | str                                 | The type of scaler to use for preprocessing the data. Possible values are "MinMaxScaler", "StandardScaler", and "RobustScaler".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | "StandardScaler"                  |
| `use_kfold`                 | bool                                | Whether to use k-fold cross-validation technique or not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | False                             |
| `test_size`                 | float                               | The proportion of the dataset to include in the test split. Should be a value between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 0.2                               |
| `val_size`                  | float                               | The proportion of the training set to use for validation. Should be a value between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 0.2                               |
| `save_models_as_dot_format` | bool                                | Whether to save the models as a dot format file. If set to True, then you should have graphviz software installed on your machine.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | False                             |
| `remove_overlapping_events` | bool                                | Whether to remove the overlapping events or not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | True                              |
| `dropout`                   | float                               | The dropout rate, which determines the fraction of input units to drop during training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0.3                               |
| `last_act_func`             | str                                 | Activation function for the final layer of each model. If set to `None`, no activation will be applied (i.e., "linear" activation: `a(x) = x`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | "sigmoid"                         |

#### Prepare data for computing the overlapping parameter `op`
The second thing to do is to prepare the events and the dataset for computing `op`:
```python
meta_model.prepare_data_and_computing_op()
```

#### Stacking Ensemble Learning Pipeline
The third thing to do is to build a stacking learning pipeline using the provided models and hyperparameters:

```python
meta_model.build_stacking_learning()
```

#### Event Extraction Optimization
The fourth thing to do is to run the Event Extraction Optimization process:

```python
meta_model.event_extraction_optimization()
```

#### Get The Results and Plots
Finally, you can plot the results, which are saved automatically: losses, true/predicted ops, true/predicted events, and delta_t.

```python
meta_model.plot_save(show_plots=True)
```
## How to credit our package

If you use our package, please cite the following paper:
```python
@misc{azib_renard_garnier_génot_andré_2023,
 title={Universal Event Detection in Time Series},
 url={osf.io/uabjg},
 DOI={10.31219/osf.io/uabjg},
 publisher={OSF Preprints},
 author={Azib, Menouar and Renard, Benjamin and Garnier, Philippe and Génot, Vincent and André, Nicolas},
 year={2023},
 month={Jul}
}
```


<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/eventdetector/a4d7e137f88a4a476ba6d07f43337ec39543a522/images/logo_eventdetector.svg" width="400">
</h1><br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/eventdetector-ts.svg?color=brightgreen)](https://pypi.org/project/eventdetector-ts/)
![Unit Tests and Lint](https://github.com/menouarazib/eventdetector/actions/workflows/unit_tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/menouarazib/eventdetector/badge.svg?branch=master)](https://coveralls.io/github/menouarazib/eventdetector?branch=master)
[![License](https://img.shields.io/github/license/menouarazib/eventdetector)](https://github.com/menouarazib/eventdetector/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.org/2310.16485.svg)](https://doi.org/10.48550/arXiv.2310.16485)

Universal Event Detection in Time Series
==========================================================
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quickstart](#quickstart-examples)
- [Make Prediction](#make-prediction)
- [Documentation](#documentation)
- [How to credit our package](#how-to-credit-our-package)
- [Futures Works](#future-works)
- [References](#references)


## Introduction
We present a new `Universal` deep-learning supervised method for detecting events in
multivariate time series data. This method combines `4` distinct
novelties compared to existing deep-learning supervised methods.
Firstly, it is based on `regression` instead of binary classification.
Secondly, it `does not require labeled datasets` where each point is
labeled; instead, it only requires reference events defined as time
points or intervals of time. Thirdly, it is designed to be `robust`
through the use of a stacked ensemble learning metamodel that
combines deep learning models, from classic Feed-Forward
Neural Networks (FFNs) to the state-of-the-art architectures like
Transformers. By leveraging the collective strengths of multiple
models, this ensemble approach can mitigate individual model
weaknesses and biases, resulting in more robust predictions.
Finally, to facilitate practical implementation, we have developed
this package called `EventDetector` to accompany the proposed method. 
It provides a rich support for event detection in time series. We establish `mathematically` that our method is universal, and capable of detecting any type of event with arbitrary precision under mild continuity assumptions on the time series. These events may encompass change points, frauds, anomalies, physical occurrences, and more. We substantiate our theoretical results using the universal approximation theorem. Additionally, we provide empirical validations that confirm our claims, demonstrating that our method, with a limited number of parameters, outperforms other deep learning approaches, particularly for rare events and imbalanced datasets from different domains.

## Installation

**Before installing this package, please ensure that you have `TensorFlow` installed in your environment.** This package relies on `TensorFlow` for its functionality, but does not include it as a dependency to allow users to manage their own TensorFlow installations. You can install TensorFlow via pip with `pip install tensorflow`.

Once TensorFlow is installed, you can proceed with the installation of this package.
Please follow the instructions below:
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

## Quickstart Examples
### Data sets
To quickly get started with `EventDetector`, follow the steps below:

- You can either download the datasets and event catalogs manually or use the built-in methods for the desired application:
  - Bow Shock Crossings: `eventdetector_ts.load_martian_bow_shock()`
      - [bow_shock_dataset](https://archive.org/download/martian_bow_shock_dataset/martian_bow_shock_dataset.pkl)
      - [bow_shock_events](https://archive.org/download/martian_bow_shock_events/martian_bow_shock_events.csv)
  - Credit Card Frauds: `eventdetector_ts.load_credit_card_fraud()`
      - [credit_card_dataset](https://archive.org/download/credit_card_fraud_dataset/credit_card_fraud_dataset.csv)
      - [credit_card_events](https://archive.org/download/credit_card_fraud_events/credit_card_fraud_events.csv)
  - NLP:
      - [Keyword extraction and the identification of tags for part-of-speech tagging (POS) in textual data.](https://github.com/menouarazib/InformationRetrievalInNLP)

### Code Implementations:
  - Credit Card Frauds:
```python
from eventdetector_ts import load_credit_card_fraud, FFN
from eventdetector_ts.metamodel.meta_model import MetaModel

dataset, events = load_credit_card_fraud()

meta_model = MetaModel(dataset=dataset, events=events, width=2, step=1,
                       output_dir='credit_card_fraud', batch_size=3200, s_h=0.01, models=[(FFN, 1)],
                       hyperparams_ffn=(1, 1, 20, 20, "sigmoid"))

meta_model.fit()

```
  - Martian Bow Shock:
```python
from eventdetector_ts import load_martian_bow_shock, FFN
from eventdetector_ts.metamodel.meta_model import MetaModel

dataset, events = load_martian_bow_shock()

meta_model = MetaModel(output_dir="mex_bow_shocks", dataset=dataset, events=events, width=76, step=1,
                       time_window=5400.0, batch_size=3000, models=[(FFN, 1)],
                       hyperparams_ffn=(1 , 1, 20, 20, "sigmoid"))

meta_model.fit()

``` 

### Performance Evaluation and Outputs

#### Comparison of Our Method with Deep Learning Methods

##### Credit Card Frauds

| Method              | Number of Parameters | Precision | Recall | F1-Score |
|---------------------|----------------------|-----------|--------|----------|
| CNN [[1]](#1)       | 119,457              | 0.89      | 0.68   | 0.77     |
| FFN+SMOTE [[2]](#2) | 5,561                | 0.79      | 0.81   | 0.80     |
| FFN+SMOTE [[3]](#3) | N/A                  | 0.82      | 0.79   | 0.81     |
| Ours                | 1,201                | 0.98      | 0.74   | 0.85     |

##### Bow Shock Crossings

| Method             | Number of Parameters | Precision | Recall        | F1-Score      |
|--------------------|----------------------|-----------|---------------|---------------|
| ResNat18 [[4]](#4) | 29,886,979           | 0.99      | [0.83 , 0.88] | [0.91 , 0.94] |
| Ours               | 6,121                | 0.95      | 0.96          | 0.95          |

#### Training and Validation Losses

The Figure below showcases the training loss and validation loss of the FFNs on the Bow Shock Crossings and Credit Card Frauds.
The low losses observed in both cases indicate that the metamodel has successfully learned the underlying patterns,
justifying the obtained good metrics.

<p align="center">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/losses_ccf.png" width="400" alt="Training and Validation Losses for Credit Card Frauds">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/losses_bs.png" width="400" alt="Training and Validation Losses for Bow Shock Crossings">
</p>

#### Comparison of Predicted `op` and True `op`
The Figure below illustrates the comparison between the predicted $op$ values and the true $op$ values on the Bow Shock Crossings and Credit Card Frauds.
<p align="center">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/op_ccf.png" width="400" height="400" alt="Predicted $op$ for Credit Card Frauds">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/op_bs.png" width="400" height="400" alt="Predicted $op$ for Bow Shock Crossings">
</p>

#### Distribution of time differences δ(t) between predicted events and ground truth events for Bow Shock Crossings and Credit Card Frauds
<p align="center">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/delta_t_ccf.png" width="400" alt="Predicted $op$ for Credit Card Frauds">
  <img src="https://raw.githubusercontent.com/menouarazib/eventdetector/master/images/delta_t_bs.png" width="400" alt="Predicted $op$ for Bow Shock Crossings">
</p>


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
For a deeper understanding of the parameters presented below,
please refer to our paper available at this [link](https://osf.io/uabjg).

### Meta Model
The first step is to instantiate the `MetaModel` object with the required arguments:
```python
from eventdetector_ts.metamodel.meta_model import MetaModel

meta_model = MetaModel(output_dir=..., dataset=..., events=..., width=..., step=...)
```
For a complete description of the required and optional arguments, please refer to the following tables:

#### Required Arguments
| Argument       | Type                      | Description                                                                                                                                                                                                                                          | Default Value |
|----------------|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `output_dir`   | str                       | The name or path of the directory where all outputs will be saved. If `output_dir` is a folder name, the full path in the current directory will be created.                                                                                         | -             |
| `dataset`      | pd.DataFrame              | The input dataset as a Pandas DataFrame.                                                                                                                                                                                                             | -             |
| `events`       | Union[list, pd.DataFrame] | The input events as either a list or a Pandas DataFrame.                                                                                                                                                                                             | -             |
| `width`        | int                       | Number of consecutive time steps in each partition (window) when creating overlapping partitions (sliding windows).                                                                                                                                  | -             |
| `step`         | int                       | Number of time steps to advance the sliding window.                                                                                                                                                                                                  | 1             |
| `width_events` | Union[int, float]         | The width of each event. If it's an `ìnt`, it represents the number of time steps that constitute an event. If it's a `float`, it represents the duration in seconds of each event. If not provided (None), it defaults to the value of  `width -1`. | `width -1`    |

#### Optional Arguments: Kwargs
| Argument                    | Type                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Default Value                          |
|-----------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| `t_max`                     | float                                    | The maximum total time is linked to the `sigma` variable of the Gaussian filter. This time should be expressed in the same unit of time (seconds, minutes, etc.) as used in the dataset. The unit of time for the dataset is determined by its time sampling. In other words, the `sigma` variable should align with the timescale used in your time series data.                                                                                                                                                                                                                                                                                                                                                               | (3 x `(width -1)` x time_sampling) / 2 |
| `delta`                     | Union[int, float]                        | The maximum time tolerance used to determine the correspondence between a predicted event and its actual counterpart in the true events. If it's an integer, it represents the number of time steps. If it's a float, it represents the duration in seconds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `width_events` x time_sampling         |
| `s_h`                       | float                                    | A step parameter for adjusting the peak height threshold `h` during the peak detection process.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 0.05                                   |
| `epsilon`                   | float                                    | A small constant used to control the size of set which contains the top models with the lowest MSE values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 0.0002                                 |
| `pa`                        | int                                      | The patience for the early stopping algorithm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 5                                      |
| `t_r`                       | float                                    | The ratio threshold for the early stopping algorithm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 0.97                                   |
| `time_window`               | Union[int, float]                        | This parameter controls the amount of data within the dataset is used for the training process. If it's an integer, it represents a specific number time steps.  If it's a float, it represents a duration in seconds. By default, it is set to None, which means all available data will be used. However, if a value is provided, the dataset will include a specific interval of data surrounding each reference event. This interval includes data from both sides of each event, with a duration equal to the specified `time_window`. Setting a `time_window` in some situations can offer several advantages, such as accelerating the training process and enhancing the neural networks' understanding of rare events. | None                                   |
| `models`                    | List[Union[str, Tuple[str, int]]]        | Determines the type of deep learning models and the number of instances to use. Available models: `LSTM`, `GRU`, `CNN`, `RNN_BIDIRECTIONAL`, `RNN_ENCODER_DECODER`, `CNN_RNN`, `FFN`, `CONV_LSTM1D`, `SELF_ATTENTION`, `TRANSFORMER`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `[(FFN, 2)]`                           |
| `hyperparams_ffn`           | Tuple[int, int, int, int, str]           | Specify for the FFN the minimum and the maximum number of layers, the minimum and the maximum number of neurons per layer, and the activation function. The List of available activation functions are ["relu","sigmoid","tanh","softmax","leaky_relu","elu","selu","swish"]. If you pass `None`, no activation is applied (i.e. "linear" activation: `a(x) = x`).                                                                                                                                                                                                                                                                                                                                                              | (1, 3, 64, 256, "sigmoid")             |
| `hyperparams_cnn`           | Tuple[int, int, int, int, int, int, str] | Specify for the CNN the minimum and maximum number of filters, the minimum and the maximum kernel size, the minimum and maximum number of pooling layers, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | (16, 64, 3, 8 , 1, 2, "relu")          |
| `hyperparams_transformer`   | Tuple[int, int, int, bool, str]          | Specify for Transformer the Key dimension, number of heads, the number of the encoder blocks, a flag to indicate the use of the original architecture, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | (256, 8, 10, True, "relu")             |
| `hyperparams_rnn`           | Tuple[int, int, int, int, str]           | Specify for the RNN the minimum and maximum number of recurrent layers,the minimum and the maximum number of hidden units, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (1,2, 16, 128,"tanh")                  |
| `hyperparams_mm_network`    | Tuple[int,int,str]                       | Specify for the MetaModel network the number of layers,the number of neurons per layer, and the activation function.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | (1 ,32,"sigmoid")                      |
| `epochs`                    | int                                      | The number of epochs to train different models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 256                                    |
| `batch_size`                | int                                      | The number of samples per gradient update.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 32                                     |
| `fill_nan`                  | str                                      | Specifies the method to use for filling `NaN` values in the dataset. Supported methods are 'zeros', 'ffill', 'bfill', and 'median'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | "zeros"                                |
| `type_training`             | str                                      | Specifies the type of training technique to use for the MetaModel. Supported techniques are 'average' and 'ffn'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | "average"                              |
| `scaler`                    | str                                      | The type of scaler to use for preprocessing the data. Possible values are "MinMaxScaler", "StandardScaler", and "RobustScaler".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | "StandardScaler"                       |
| `use_kfold`                 | bool                                     | Whether to use k-fold cross-validation technique or not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | False                                  |
| `test_size`                 | float                                    | The proportion of the dataset to include in the test split. Should be a value between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 0.2                                    |
| `val_size`                  | float                                    | The proportion of the training set to use for validation. Should be a value between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 0.2                                    |
| `save_models_as_dot_format` | bool                                     | Whether to save the models as a dot format file. If set to True, then you should have graphviz software installed on your machine.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | False                                  |
| `remove_overlapping_events` | bool                                     | Whether to remove the overlapping events or not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | True                                   |
| `dropout`                   | float                                    | The dropout rate, which determines the fraction of input units to drop during training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0.3                                    |
| `last_act_func`             | str                                      | Activation function for the final layer of each model. If set to `None`, no activation will be applied (i.e., "linear" activation: `a(x) = x`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | "sigmoid"                              |

#### The method `fit`
The method `fit` calls automatically the following methods: 
##### Prepare data for computing the overlapping parameter `op`
The second thing to do is to prepare the events and the dataset for computing `op`:
```python
meta_model.prepare_data_and_computing_op()
```

##### Stacking Ensemble Learning Pipeline
The third thing to do is to build a stacking learning pipeline using the provided models and hyperparameters:

```python
meta_model.build_stacking_learning()
```

##### Event Extraction Optimization
The fourth thing to do is to run the Event Extraction Optimization process:

```python
meta_model.event_extraction_optimization()
```

##### Get The Results and Plots
Finally, you can plot the results, which are saved automatically: losses, true/predicted ops, true/predicted events, and delta_t.

```python
meta_model.plot_save(show_plots=True)
```
## How to credit our package

If you use our package, please cite the following papers:

```bash
@misc{azib2023universal,
      title={Universal Event Detection in Time Series}, 
      author={Menouar Azib and Benjamin Renard and Philippe Garnier and Vincent Génot and Nicolas André},
      year={2023},
      eprint={2311.15654},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

```bash
@misc{azib2023comprehensive,
      title={A Comprehensive Python Library for Deep Learning-Based Event Detection in Multivariate Time Series Data and Information Retrieval in NLP}, 
      author={Menouar Azib and Benjamin Renard and Philippe Garnier and Vincent Génot and Nicolas André},
      year={2023},
      eprint={2310.16485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Future Works
In our future works, we aim to enhance our model’s capabilities by predicting events of varying durations. This would be a significant improvement over our current approach, which only predicts the midpoint of events with a fixed duration.

# References

<a id="3"> [1] F. K. Alarfaj, I. Malik, H. U. Khan, N. Almusallam, M. Ramzan and M. Ahmed, “Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms,” in IEEE Access, vol. 10, pp. 39700-39715, 2022, doi: 10.1109/ACCESS.2022.3166891.
</a>

<a id="4"> [2] D. Varmedja, M. Karanovic, S. Sladojevic, M. Arsenovic and A. Anderla, “Credit Card Fraud Detection - Machine Learning methods,” 2019 18th International Symposium INFOTEH-JAHORINA (INFOTEH), East Sarajevo, Bosnia and Herzegovina, 2019, pp. 1-5, doi: 10.1109/INFOTEH.2019.8717766.
</a>

<a id="5"> [3] E. Ileberi, Y. Sun and Z. Wang, “A machine learning based credit card fraud detection using the GA algorithm for feature selection,” in J Big Data, vol. 9, no. 24, 2022. [Online]. Available: https://doi.org/10.1186/s40537-022-00573-8.
</a>

<a id="6"> [4] I. K. Cheng, N. Achilleos and A. Smith, “Automated bow shock and magnetopause boundary detection with Cassini using threshold and deep learning methods,” Front. Astron. Space Sci., vol. 9, 2022, doi: 10.3389/fspas.2022.1016453.
</a>


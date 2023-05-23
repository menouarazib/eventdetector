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

from eventdetector import MIDDLE_EVENT_LABEL, FFN
from eventdetector.metamodel.meta_model import MetaModel
from eventdetector.prediction.prediction import predict
from eventdetector.prediction.utils import plot_prediction

# Get the dataset.
dataset_mex: pd.DataFrame = pd.read_pickle("mex_dataset_2012.pkl")
start_date = datetime(2012, 1, 1)
stop_date = datetime(2012, 2, 1)
# Filtering dataset by giving a starting date and an ending date.
dataset_mex_learning = dataset_mex[(dataset_mex.index >= start_date) & (dataset_mex.index <= stop_date)]
print(dataset_mex_learning)
# Get the events.
mex_bow_shocks: pd.DataFrame = pd.read_pickle("mex_events.pkl")
# Filtering events by giving a starting date and an ending date.
mex_bow_shocks = mex_bow_shocks[
    (mex_bow_shocks[MIDDLE_EVENT_LABEL] >= start_date) & (mex_bow_shocks[MIDDLE_EVENT_LABEL] <= stop_date)]

# This parameter determines the amount of data to include in the dataset around each reference event, specified in
# units of time. By default, it is set to None, and in this case all data will be used.
time_window: int = 5400

# Create the MetaModel
meta_model = MetaModel(output_dir="mex_bow_shocks", dataset=dataset_mex_learning, events=mex_bow_shocks,
                       width=45, step=1, time_window=time_window, batch_size=3000, models=[(FFN, 1)])

# Prepare the events and dataset for computing op.
meta_model.prepare_data_and_computing_op()
# Builds a stacking learning pipeline using the provided models and hyperparameters.
meta_model.build_stacking_learning()
# Run the Event Extraction Optimization process.
meta_model.event_extraction_optimization()
# Plot the results: true/predicted op, true/predicted events, deltat_t.
meta_model.plot()
``

5. Make Predictions
```python
# Make Predictions
start_date_prediction = datetime(2012, 10, 1)
stop_date_prediction = datetime(2013, 1, 1)

dataset_mex_prediction = dataset_mex[
    (dataset_mex.index >= start_date_prediction) & (dataset_mex.index <= stop_date_prediction)]
print(dataset_mex_prediction)

# Call the 'predict' method
predicted_events, predicted_op, filtered_predicted_op = predict(dataset=dataset_mex,
                                                                path='path/mex_bow_shocks/')
# Plot the predicted Op
plot_prediction(predicted_op=predicted_op, filtered_predicted_op=filtered_predicted_op)
``
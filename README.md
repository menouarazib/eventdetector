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

from eventdetector import MIDDLE_EVENT_LABEL
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
    (events[MIDDLE_EVENT_LABEL] >= start_date) & (events[MIDDLE_EVENT_LABEL] <= stop_date)]

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
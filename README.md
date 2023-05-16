<h1>Universal Approximation for Event Detection in Time Series</h1>
<p>Welcome to <strong>Event Detector</strong>, a Python package for detecting events in time series data.
The emphasis of this package is on offering useful machine learning functionalities, such as customizing and fitting the model on multidimensional time series, training on large datasets,
ensembling models, and providing rich support for event detection in time series. To get started using <strong>Event Detector</strong>, 
simply follow the instructions below to install the required packages and dependencies.</p>
<ol>
    <li>Clone the repository:</li>
</ol>
<pre><code>git clone https://github.com/menouarazib/eventdetector.git
cd eventdetector
</code></pre>
<ol start="2">
    <li>Create a virtual environment:</li>
</ol>
<pre><code>python -m venv env
source env/bin/activate  # for Linux/MacOS
env\Scripts\activate.bat  # for Windows
</code></pre>
<ol start="3">
    <li>Install the required packages:</li>
</ol>
<pre><code>pip install -r requirements.txt</code></pre>
<ol start="4">
 <li>Quickstart Example</li>
    <p>To quickly get started with the Event Detection in Time Series package, follow the steps below:</p>
    <p>Download the dataset and events using the links provided:</p>
    <ul>
        <li><a href="https://drive.google.com/file/d/1v8W50aveNMUeofDOQoI_601E0IN990BS/view?usp=sharing">mex_dataset_2012</a></li>
        <li><a href="https://drive.google.com/file/d/1cMZn4fsgot2J2EffNCKvm0I2XKiIemkl/view?usp=sharing">mex_events</a></li>
    </ul>
</ol>

<pre><code>
from datetime import datetime

import pandas as pd

from eventdetector import MIDDLE_EVENT_LABEL, FFN, GRU
from eventdetector.metamodel.meta_model import MetaModel

# Get the dataset.
dataset_mex: pd.DataFrame = pd.read_pickle("mex_dataset_2012.pkl")
start_date = datetime(2012, 1, 1)
stop_date = datetime(2012, 5, 1)
# Filtering dataset by giving a starting date and an ending date.
dataset_mex = dataset_mex[(dataset_mex.index >= start_date) & (dataset_mex.index <= stop_date)]
print(dataset_mex)
# Get the events.
mex_bow_shocks: pd.DataFrame = pd.read_pickle("mex_events.pkl")
# Filtering events by giving a starting date and an ending date.
mex_bow_shocks = mex_bow_shocks[
    (mex_bow_shocks[MIDDLE_EVENT_LABEL] >= start_date) & (mex_bow_shocks[MIDDLE_EVENT_LABEL] <= stop_date)]
# This parameter determines the amount of data to include in the dataset around each reference event, specified in
# units of time.
time_window: int = 5400  # in seconds
# Create the MetaModel
meta_model = MetaModel(output_dir="mex_bow_shocks", dataset=dataset_mex, events=mex_bow_shocks, width=45, step=1,
                       time_window=time_window, batch_size=3000, models=[(FFN, 2), (GRU, 1)])
# Prepare the events and dataset for computing op.
meta_model.prepare_data_and_computing_op()
# Builds a stacking learning pipeline using the provided models and hyperparameters.
meta_model.build_stacking_learning()
# Run the Event Extraction Optimization process.
meta_model.event_extraction_optimization()
# Plot the results: true/predicted op, true/predicted events, deltat_t.
meta_model.plot()</code></pre>
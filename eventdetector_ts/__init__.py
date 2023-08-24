import os
from enum import Enum
from logging import config
from typing import Dict, Optional
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm

TIME_LABEL = "time"
MIDDLE_EVENT_LABEL = "event"

LSTM = "LSTM"
GRU = "GRU"
CNN = "CNN"
RNN_BIDIRECTIONAL = "RNN_BIDIRECTIONAL"
CONV_LSTM1D = "CONV_LSTM_1D"
RNN_ENCODER_DECODER = "RNN_ENCODER_DECODER"
CNN_RNN = "CNN_RNN"
SELF_ATTENTION = "SELF_ATTENTION"
TRANSFORMER = "TRANSFORMER"
FFN = "FFN"

FILL_NAN_ZEROS = 'zeros'
FILL_NAN_FFILL = 'ffill'
FILL_NAN_BFILL = 'bfill'
FILL_NAN_MEDIAN = 'median'

TYPE_TRAINING_AVERAGE = 'average'
TYPE_TRAINING_FFN = 'ffn'
META_MODEL_NETWORK = "meta_model_ffn"
META_MODEL_SCALER = "meta_model_scaler"

# Define constants for scaler types
MIN_MAX_SCALER = "MinMaxScaler"
STANDARD_SCALER = "StandardScaler"
ROBUST_SCALER = "RobustScaler"

SCALERS_DIR = "scalers"
MODELS_DIR = "models"
OUTPUT_DIR = "output"
CONFIG_FILE = ".config.json"
# Store some important values for prediction
config_dict: Dict = {}


class TimeUnit(Enum):
    """
    An enumeration of different time units.

    Attributes:
        SECOND: The time unit is in seconds.
        MILLISECOND: The time unit is in milliseconds.
        MICROSECOND: The time unit is in microseconds.
        MINUTE: The time unit is in minutes.
        HOUR: The time unit is in hours.
        DAY: The time unit is in days.
        YEAR: The time unit is in years.
    """
    SECOND = "second"
    MILLISECOND = "millisecond"
    MICROSECOND = "microsecond"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    YEAR = "year"

    @classmethod
    def _missing_(cls, value):
        return cls.SECOND

    def __str__(self):
        return self.value


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(asctime)s %(log_color)s[%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

config.dictConfig(LOGGING_CONFIG)


def my_hook(t):
    """
    Wraps tqdm instance. Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
  
    Example
    -------
  
    
  
    """
    last_b = [0]

    def inner(b=1, bsize=1, t_size=None):
        """
        b  : int, optional
            Number of blocks just transferred [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        t_size  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if t_size is not None:
            t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def load_dataset(file_path: str, name: str, url=None, index_col: Optional[int] = 0) -> pd.DataFrame:
    """
    Load a dataset from a file. If the file is not found, it will be downloaded from the given URL.

    Args:
        name: Name of the file to load
        index_col: the same value as pandas index_col
        file_path (str): The path to the dataset file.
        url (str): The URL from which to download the dataset (optional).

    Returns:
        pandas.DataFrame: The loaded dataset.
    """

    file_extension = os.path.splitext(file_path)[1].lower()

    if not os.path.isfile(file_path) and url:
        # Dataset file isn't found, download it
        with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,
                  desc=f"Downloading {name}") as t:  # all optional kwargs
            urlretrieve(url, filename=file_path,
                        reporthook=my_hook(t), data=None)

    if file_extension == ".csv":
        # Read CSV file
        dataset = pd.read_csv(file_path, index_col=index_col)
    elif file_extension == ".pkl":
        # Read Pickle file
        dataset = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Return the loaded dataset
    return dataset


def load_martian_bow_shock():
    """
        Load the Martian bow shock dataset and events, for more information check this link:  http://amda.cdpp.eu/

        Returns:
            A dataset and events as pd.DataFrame

        """
    url_dataset = "https://archive.org/download/martian_bow_shock_dataset/martian_bow_shock_dataset.pkl"
    url_events = "https://archive.org/download/martian_bow_shock_events/martian_bow_shock_events.csv"
    data_set = load_dataset(file_path="martian_bow_shock_dataset.pkl", name="Martian Bow Shock data set",
                            url=url_dataset)
    events = load_dataset(file_path="martian_bow_shock_events.csv", name="Martian Bow Shock events", index_col=None,
                          url=url_events)

    return data_set, events


def load_credit_card_fraud():
    """
    Load the credit card fraud dataset and events, for more information check this link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Returns:
        A dataset and events as pd.DataFrame

    """
    url_dataset = "https://archive.org/download/credit_card_fraud_dataset/credit_card_fraud_dataset.csv"
    url_events = "https://archive.org/download/credit_card_fraud_events/credit_card_fraud_events.csv"

    data_set = load_dataset(file_path="credit_card_fraud_dataset.csv", name="Credit Card Fraud data set",
                            url=url_dataset)
    events = load_dataset(file_path="credit_card_fraud_events.csv", name="Credit Card Fraud events", index_col=None,
                          url=url_events)

    return data_set, events

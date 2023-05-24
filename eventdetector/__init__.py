from enum import Enum
from logging import config
from typing import Dict

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

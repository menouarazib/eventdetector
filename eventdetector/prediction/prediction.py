import json
import os
from typing import Dict, Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from eventdetector import CONFIG_FILE, SCALERS_DIR, TYPE_TRAINING_FFN, TimeUnit, MODELS_DIR
from eventdetector.data.helpers import convert_dataframe_to_sliding_windows, get_timedelta
from eventdetector.optimization.algorithms import convolve_with_gaussian_kernel
from eventdetector.optimization.event_extraction_pipeline import get_peaks
from eventdetector.prediction import logger


def load_config_file(path: Optional[str] = None) -> Dict:
    # Load config file of the meta-model
    current_directory = os.path.abspath(".")
    if path is not None:
        current_directory = path

    config_file_path = os.path.join(current_directory, CONFIG_FILE)
    if not os.path.exists(config_file_path):
        msg: str = f"The config file {CONFIG_FILE} does not exist in this path: {config_file_path}"
        logger.critical(msg)
        raise ValueError(msg)

    with open(config_file_path, 'r') as f:
        config_: Dict = json.load(f)
        return config_


def load_models(model_keys: List[str], output_dir: str) -> List[tf.keras.Model]:
    models: List[tf.keras.Model] = []
    for key in model_keys:
        path = os.path.join(output_dir, MODELS_DIR)
        path = os.path.join(path, key)
        models.append(tf.keras.models.load_model(path))
    return models


def apply_scaling(x: np.ndarray, config_data: Dict) -> np.ndarray:
    n_time_steps = x.shape[1]
    output_dir: str = config_data.get("output_dir")
    scalers_dir = os.path.join(output_dir, SCALERS_DIR)
    try:
        for i in range(n_time_steps):
            scaler_i_path = os.path.join(scalers_dir, f'scaler_{i}.joblib')
            # Print progress
            print("\rLoading scaling...{}/{}".format(i + 1, n_time_steps), end="")
            # Load the scaler from disk
            scaler = joblib.load(scaler_i_path)
            x[:, i, :] = scaler.transform(x[:, i, :])
    except ValueError as e:
        logger.critical(e)
        raise e

    return np.asarray(x).astype('float32')


def compute_op_as_mid_times(sliding_windows: np.ndarray, op_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = []
    op_g_ = []
    for n in range(len(op_g)):
        w_n = sliding_windows[n]
        b_n = w_n[0][-1].to_pydatetime()
        e_n = w_n[-1][-1].to_pydatetime()
        c_n = b_n + (e_n - b_n) / 2
        t.append(c_n)
        op_g_.append(op_g[n])
    t, op_g_ = np.array(t), np.array(op_g_)
    return t, op_g_


def predict(dataset: pd.DataFrame, path: Optional[str] = None) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Generates output predictions for the input dataset
    Args:
        dataset (pd.DataFrame): The input dataset
        path (Optional[str] = None): The path to the created folder by the MetaModel 

    Returns:
        Tuple[List, np.ndarray, np.ndarray]: Predicted events, predicted Op and filtered predicted Op
    """
    config_data: Dict = load_config_file(path=path)
    logger.info(f"Config dict: {config_data}")
    logger.info("Converting the dataset to sliding windows.")
    dataset_as_sliding_windows: np.ndarray = convert_dataframe_to_sliding_windows(dataset,
                                                                                  width=config_data.get("width"),
                                                                                  step=config_data.get("step"),
                                                                                  fill_method=config_data.get(
                                                                                      'fill_nan'))

    # Remove the column containing the timestamps from the sliding windows
    x: np.ndarray = np.delete(dataset_as_sliding_windows, -1, axis=2)
    logger.info(f"Applying a scaling for data of shape: {x.shape}")
    x = apply_scaling(x=x, config_data=config_data)
    model_keys: List[str] = config_data.get('models')
    logger.info(f"Loading models: {model_keys}")
    models: List[tf.keras.Model] = load_models(model_keys=model_keys, output_dir=config_data.get('output_dir'))
    batch_size: int = config_data.get("batch_size")
    predictions = []
    logger.info("Making predictions")
    for model in models:
        # Make predictions using each model
        predicted_y: np.ndarray = model.predict(x, batch_size=batch_size,
                                                use_multiprocessing=True)
        predicted_y = predicted_y.flatten()
        predictions.append(predicted_y)

    logger.info("Making predictions from the MetaModel")
    type_training: str = config_data.get('type_training')
    # Convert a list of 1D NumPy arrays to 2D NumPy array
    predictions = np.stack(predictions, axis=1)
    if type_training == TYPE_TRAINING_FFN:
        # TODO
        predicted_op = np.mean(predictions, axis=1)
    else:
        predicted_op = np.mean(predictions, axis=1)

    sigma, m, h = config_data.get('best_combination')
    logger.info(f"Applying Gaussian Filer with sigma = {sigma} and m = {m}")
    filtered_predicted_op = convolve_with_gaussian_kernel(predicted_op, sigma=sigma, m=m)
    logger.info("Computing filtered predictions as a function of the mid-times of the sliding windows")
    t, filtered_predicted_op = compute_op_as_mid_times(sliding_windows=dataset_as_sliding_windows,
                                                       op_g=filtered_predicted_op)
    logger.info(f"Computing peaks with h = {h}")
    s_peaks = get_peaks(h=h, t=t, op_g=filtered_predicted_op)
    predicted_events = []
    time_unit: TimeUnit = TimeUnit.__call__(config_data.get('time_unit'))
    radius = get_timedelta(config_data.get("w_s") // 2, time_unit)
    logger.info(f"Generating a predicted events with radius = {radius}, predicted op and a filtered predicted op")
    for i in range(len(s_peaks)):
        predicted_event = s_peaks[i]
        start_time = predicted_event - radius
        end_time = predicted_event + radius
        predicted_events.append((start_time.isoformat(), end_time.isoformat()))
    return predicted_events, predicted_op, filtered_predicted_op
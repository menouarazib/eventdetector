import json
import os
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from eventdetector_ts import CONFIG_FILE, SCALERS_DIR, TYPE_TRAINING_FFN, TimeUnit, MODELS_DIR, META_MODEL_NETWORK, \
    META_MODEL_SCALER
from eventdetector_ts.data.helpers import convert_dataframe_to_overlapping_partitions, get_timedelta
from eventdetector_ts.optimization.algorithms import convolve_with_gaussian_kernel
from eventdetector_ts.optimization.event_extraction_pipeline import get_peaks, compute_op_as_mid_times
from eventdetector_ts.prediction import logger


def load_config_file(path: str) -> Dict:
    """
     Load config file of the meta-model.
     
    Args:
        path (str): Where the config file is stored

    Returns:
        Data as a Dict which contains all configuration information
    """
    config_file_path = os.path.join(path, CONFIG_FILE)
    if not os.path.exists(config_file_path):
        msg: str = f"The config file {CONFIG_FILE} does not exist in this path: {config_file_path}"
        logger.critical(msg)
        raise ValueError(msg)

    with open(config_file_path, 'r') as f:
        config_: Dict = json.load(f)
        return config_


def load_models(model_keys: List[str], output_dir: str) -> List[tf.keras.Model]:
    """
    Loads the trained models.
    Args:
        model_keys (List[str]): List of model's name
        output_dir (str): The parent directory where the trained models are stored

    Returns:
        List of keras models
    """
    models: List[tf.keras.Model] = []
    for key in model_keys:
        path = os.path.join(output_dir, MODELS_DIR)
        path = os.path.join(path, key)
        models.append(tf.keras.models.load_model(path))
    return models


def apply_scaling(x: np.ndarray, config_data: Dict) -> np.ndarray:
    """
    Scaling input data according to the stored scalers.
    Args:
        x (np.ndarray): Input data to be scaled 
        config_data (Dict): Configuration Data 

    Returns:
        Scaled data.
    """
    n_time_steps = x.shape[1]
    output_dir: str = config_data.get("output_dir")
    scalers_dir = os.path.join(output_dir, SCALERS_DIR)
    try:
        for i in range(n_time_steps):
            scaler_i_path = os.path.join(scalers_dir, f'scaler_{i}.joblib')
            # Print progress
            print("\rLoading and applying scalers...{}/{}".format(i + 1, n_time_steps), end="")
            # Load the scaler from disk
            scaler = joblib.load(scaler_i_path)
            x[:, i, :] = scaler.transform(x[:, i, :])
    except ValueError as e:
        logger.critical(e)
        raise e

    logger.info("Convert data to float32 for consistency...")
    x = np.asarray(x).astype('float32')
    return x


def load_meta_model(output_dir: str) -> Tuple[tf.keras.Model, Any]:
    """
    Load the metamodel network and the scaler.
    Args:
        output_dir (str): The parent directory where the trained models are stored

    Returns:
        tf.keras.Model, StanderScaler
    """
    path = os.path.join(output_dir, MODELS_DIR)
    path = os.path.join(path, META_MODEL_NETWORK)
    model = tf.keras.models.load_model(path)
    scalers_dir = os.path.join(output_dir, SCALERS_DIR)
    scaler_path = os.path.join(scalers_dir, f'{META_MODEL_SCALER}.joblib')
    scaler = joblib.load(scaler_path)

    return model, scaler


def predict(dataset: pd.DataFrame, path: str) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Generates output predictions for the input dataset
    Args:
        dataset (pd.DataFrame): The input dataset.
        path (str): The path to the created folder by the MetaModel. 

    Returns:
        Tuple[List, np.ndarray, np.ndarray]: Predicted events, predicted Op and filtered predicted Op
    """

    if path is None or not isinstance(path, str) or len(path) == 0:
        msg: str = f"The provided path {path} is not valid."
        logger.critical(msg)
        raise ValueError(msg)

    config_data: Dict = load_config_file(path=path)
    config_data['output_dir'] = path
    logger.info(f"Config dict: {config_data}")
    logger.info("Converting the dataset to overalapping partitions.")
    dataset_as_overlapping_partitions: np.ndarray = convert_dataframe_to_overlapping_partitions(dataset,
                                                                                                width=config_data.get(
                                                                                                    "width"),
                                                                                                step=config_data.get(
                                                                                                    "step"),
                                                                                                fill_method=config_data.get(
                                                                                                    'fill_nan'))
    # Remove the column containing the timestamps from the overalapping partitions
    x: np.ndarray = np.delete(dataset_as_overlapping_partitions, -1, axis=2)
    logger.info(f"The shape of the input data: {x.shape}")
    x = apply_scaling(x=x, config_data=config_data)
    model_keys: List[str] = config_data.get('models')
    logger.info(f"Loading models: {model_keys}")
    models: List[tf.keras.Model] = load_models(model_keys=model_keys, output_dir=config_data.get('output_dir'))
    batch_size: int = config_data.get("batch_size")
    predictions = []
    logger.info("Making prediction from the trained models")
    for model in models:
        # Make predictions using each model
        predicted_y: np.ndarray = model.predict(x, batch_size=batch_size,
                                                use_multiprocessing=False)
        predicted_y = predicted_y.flatten()
        predictions.append(predicted_y)

    type_training: str = config_data.get('type_training')
    # Convert a list of 1D NumPy arrays to 2D NumPy array
    predictions = np.stack(predictions, axis=1)
    if type_training == TYPE_TRAINING_FFN:
        logger.info("Loading the MetaModel and its Scaler")
        model, scaler = load_meta_model(output_dir=config_data.get('output_dir'))
        predictions = scaler.transform(predictions)
        logger.info("Make a final prediction using the network of the MetaModel")
        predicted_op = model.predict(predictions, batch_size=batch_size)
        predicted_op = predicted_op.flatten()
    else:
        logger.info("Make a final prediction by averaging")
        predicted_op = np.mean(predictions, axis=1)

    sigma, m, h = config_data.get('best_combination')
    logger.info(f"Applying Gaussian Filter with sigma = {sigma} and m = {m}")
    filtered_predicted_op = convolve_with_gaussian_kernel(predicted_op, sigma=sigma, m=m)
    logger.info("Computing filtered predictions as a function of the mid-times of the overalapping partitions")
    t, filtered_predicted_op = compute_op_as_mid_times(overlapping_partitions=dataset_as_overlapping_partitions,
                                                       op_g=filtered_predicted_op)
    logger.info(f"Computing peaks with h = {h:.2f}")
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

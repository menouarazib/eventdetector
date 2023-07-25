import logging

import numpy as np
import tensorflow as tf
from keras.utils.io_utils import print_msg
from sklearn.model_selection import KFold


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Create a custom early stopping callback that stops training when the ratio of current training loss to current
    validation loss is less than a specified ratio for a number of consecutive epochs.

    Args:
        ratio (float): Ratio to compare current train loss and current val loss against.
        patience (int): Number of epochs to wait before stopping training.
        verbose (int, optional): Verbosity level.

    Attributes:
        stopped_epoch (int or None): Last epoch index where training was stopped.
        best (float or None): Best validation loss observed so far.
        best_epoch (int or None): Index of the epoch where the best validation loss was observed.
        ratio (float): Ratio to compare current train loss and current val loss against.
        patience (int): Number of epochs to wait before stopping training.
        verbose (int): Verbosity level.
        wait (int): Number of epochs since the last time the ratio was greater than self.ratio.
        monitor_op (function): Comparison operator for the ratio.
        best_weights (np.ndarray or None): Model weights at the epoch with the best validation loss.
    """

    def __init__(self, ratio: float, patience: int, verbose: int = 1):
        super().__init__()
        self.stopped_epoch = None
        self.best = None
        self.best_epoch = None
        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.monitor_op = np.greater
        self.best_weights = None

    def on_train_begin(self, logs=None):
        """
        Initialize instance attributes.
        """
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """
        Update the best validation loss and check whether to stop training.
        """
        if logs is not None:
            if self.best_weights is None:
                self.best_weights = self.model.get_weights()

            current_val = logs.get('val_loss')  # Current validation loss
            current_train = logs.get('loss')  # Current training loss
            if current_val is None:
                logging.warning(
                    "Early stopping conditioned on metric `%s` "
                    "which is not available. Available metrics are: %s",
                    'val_loss',
                    ",".join(list(logs.keys())),
                )

            # Update the best validation loss and weights
            if self.monitor_op(self.best, current_val):
                self.best = current_val
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch

            # If the ratio of current training loss to current validation loss is greater than the specified ratio.
            if self.monitor_op(np.divide(current_train, current_val), self.ratio):
                self.wait = 0
            else:
                # Only check after the first epoch.
                if self.wait >= self.patience and epoch > 0:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.verbose > 0:
                        print_msg(
                            "Restoring model weights from "
                            "the end of the best epoch: "
                            f"{self.best_epoch + 1}."
                        )
                    self.model.set_weights(self.best_weights)
                self.wait += 1

    def on_train_end(self, logs=None):
        """
        Print a message indicating that training was stopped early.
        """
        if logs is not None:
            if self.stopped_epoch > 0 and self.verbose > 0:
                print_msg(
                    f"Epoch {self.stopped_epoch + 1}: early stopping. "
                    "Restoring model weights from "
                    "the end of the best epoch: "
                    f"{self.best_epoch + 1}. "
                    "Best validation loss: "
                    f"{self.best}."
                )


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-Attention layer for Neural Networks
    """

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__()
        self.last_attention_weights = None
        # Instantiate a multi-head attention layer with key dimensionality of units
        # and a single head
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        # Instantiate a normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization()
        # Instantiate an addition layer
        self.add = tf.keras.layers.Add()

    def call(self, query: tf.Tensor) -> tf.Tensor:
        """
        Apply a self-attention mechanism on the input query and return the output.

        Args:
            query: input tensor to the layer.

        Return:
            output tensor of the layer.
        """
        # Apply multi-head attention on a query
        attn_output, attn_scores = self.mha(
            query=query,
            key=query,
            value=query,
            return_attention_scores=True)

        # Store the attention scores in last_attention_weights for inspection
        self.last_attention_weights = attn_scores

        # Add the attention output to the query and normalize it
        x = self.add([query, attn_output])
        x = self.layer_norm(x)

        return x


def custom_cross_val_score(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, cv: KFold, epochs: int, batch_size: int,
                           callbacks: list) -> np.ndarray:
    """
    A function to perform custom cross-validation for a Keras model.
    
    Args:
        model: A Keras model.
        x: The input data.
        y: The target data.
        cv: A KFold cross-validation object.
        epochs: The number of epochs for training.
        batch_size: The batch size for training.
        callbacks: A list of Keras callbacks.

    Returns:
        The mean of the validation loss across all folds.
    """
    scores = []
    for train_index, val_index in cv.split(x):
        train_x, train_y = x[train_index], y[train_index]
        val_x, val_y = x[val_index], y[val_index]
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                            validation_data=(val_x, val_y), verbose=0)
        scores.append(np.min(history.history['val_loss']))
    return np.mean(scores)

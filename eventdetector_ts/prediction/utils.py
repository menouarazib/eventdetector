import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from eventdetector_ts.plotter import COLOR_PREDICTED, COLOR_TRUE


def plot_prediction(predicted_op: np.ndarray, filtered_predicted_op: np.ndarray) -> None:
    """
    Plot the original and filtered predicted Op
    Args:
        predicted_op (np.ndarray): Predicted Op
        filtered_predicted_op (np.ndarray): Filtered predicted Op

    Returns:
        None
    """
    sns.set(style="ticks", palette="Set2")
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Plot the true and predicted values using Seaborn
    n = len(predicted_op)
    sns.lineplot(x=np.arange(n), y=predicted_op, color=COLOR_TRUE, label='Predicted Op')
    sns.lineplot(x=np.arange(n), y=filtered_predicted_op, color=COLOR_PREDICTED, label='Filtered Predicted Op')

    # Add labels and title to the plot
    plt.xlabel('partitions')
    plt.ylabel('Op')
    plt.title('Predicted Op')
    # Add legend
    plt.legend()
    # Show
    plt.show()

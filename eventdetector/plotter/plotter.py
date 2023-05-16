import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from eventdetector import OUTPUT_DIR, TimeUnit, MIDDLE_EVENT_LABEL
from eventdetector.plotter import logger, COLOR_TRUE, COLOR_PREDICTED, STYLE_PREDICTED, STYLE_TRUE
from eventdetector.plotter.helpers import event_to_rectangle


class Plotter:
    """
    The Plotter class is responsible for generating and saving plots of the predicted and true op, events, delta_t,...
        It provides a convenient way to visualize and compare the performance of a
        predictive model against the actual observed values.
    """

    def __init__(self, root_dir: str, time_unit: TimeUnit, w_s: int, show: bool = False) -> None:
        """
        Initialize the Plotter object.

        Args:
            root_dir (str): The root directory for saving the plots.
            time_unit (TimeUnit): The unit time of the dataset
            w_s (int): The width of each event in time unit
            show (bool, optional): Whether to display the plots or not. Defaults to False.
        """
        self.w_s = w_s
        self.time_unit = time_unit
        self.show = show
        self.root_dir = root_dir
        self.predicted_y: np.ndarray = np.empty(shape=(0,))
        self.test_y: np.ndarray = np.empty(shape=(0,))
        self.predicted_events: list = []
        self.true_events: pd.DataFrame = pd.DataFrame()
        self.delta_t: list = []
        self.working_dir = os.path.join(root_dir, OUTPUT_DIR)
        os.makedirs(self.working_dir)

    def set_data_op(self, test_y: np.ndarray, predicted_y: np.ndarray) -> None:
        """
        Set test_y and predicted_y
        Args:
            test_y: The true op values
            predicted_y: The predicted op values

        Returns:
            None
        """
        self.test_y = test_y
        self.predicted_y = predicted_y

    def set_data_events(self, predicted_events: list, true_events: pd.DataFrame) -> None:
        """
        Set true and predicted events
        Args:
            predicted_events (list): List of predicted events computed by the optimization process
            true_events (pd.DataFrame): DataFrame of true events

        Returns:
            None
        """
        self.predicted_events = predicted_events
        self.true_events = true_events

    def set_delta_t(self, delta_t: list) -> None:
        """
        Set delta_t
        Args:
            delta_t (list): Each item of this list contains the accepted delta in time unit between
                true event its correspondent in the list of predicted events

        Returns:
            None
        """
        self.delta_t = delta_t

    def plot_prediction(self) -> None:
        """
        Plot the true and the predicted op and save it.

        Returns:
            None
        """

        logger.info("Plotting and saving the figure displaying the true and the predicted op")
        # Create the plot using Seaborn
        # Set the ggplot style
        sns.set(style="ticks", palette="Set2")
        plt.figure(figsize=(8, 6))  # Set the figure size
        # Plot the true and predicted values using Seaborn
        n = len(self.test_y)
        sns.lineplot(x=np.arange(n), y=self.test_y, color=COLOR_TRUE, label='True Op')
        sns.lineplot(x=np.arange(n), y=self.predicted_y, color=COLOR_PREDICTED, label='Predicted Op')
        # Add labels and title to the plot
        plt.xlabel('Windows')
        plt.ylabel('Op')
        plt.title('True Op vs Predicted Op')
        # Add legend
        plt.legend()
        # Save the plot to a file
        path = os.path.join(self.working_dir, "true_op_vs_predicted_op.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()

    def plot_predicted_events(self) -> None:
        """
        Plot the true and the predicted events and save it.

        Returns:
            None
        """

        logger.info("Plotting and saving the figure displaying the true events and the predicted events")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(style="ticks", palette="Set2")

        for i, predicted_event in enumerate(self.predicted_events):
            rect1 = event_to_rectangle(event=predicted_event, w_s=self.w_s, time_unit=self.time_unit,
                                       color=COLOR_PREDICTED,
                                       style=STYLE_PREDICTED)
            ax.add_patch(rect1)

        for _, test_date in self.true_events[MIDDLE_EVENT_LABEL].iteritems():
            rect1 = event_to_rectangle(event=test_date, w_s=self.w_s, time_unit=self.time_unit, color=COLOR_TRUE,
                                       style=STYLE_TRUE)
            ax.add_patch(rect1)

        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        start_time = self.true_events[MIDDLE_EVENT_LABEL].iloc[0]
        end_time = self.true_events[MIDDLE_EVENT_LABEL].iloc[-1]
        ax.set_xlim([start_time, end_time])
        ax.set_ylim([0.0, 1.01])

        predicted_patch = Patch(color=COLOR_PREDICTED, label='Predicted Events')
        true_patch = Patch(color=COLOR_TRUE, label='True Events')
        ax.legend(handles=[predicted_patch, true_patch], edgecolor="black")

        # Save the plot to a file
        path = os.path.join(self.working_dir, "true_events_vs_predicted_events.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()

    def plot_delta_t(self, bins=10) -> None:
        """
        Plots a histogram for delta t.

        Args:
            bins (int): The number of bins in the histogram. Default is 10.

        Returns:
              None
        """
        sns.set(style="ticks", palette="Set2")
        plt.figure(figsize=(8, 6))
        plt.hist(self.delta_t, bins=bins)
        plt.xlabel(f'delta ({self.time_unit})')
        plt.ylabel('Number of events')
        std = np.std(self.delta_t)
        mu = np.mean(self.delta_t)
        plt.title(f'Histogram std = {std:.2f}, mu = {mu:.2f}')
        # Save the plot to a file
        path = os.path.join(self.working_dir, "delta_t.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()

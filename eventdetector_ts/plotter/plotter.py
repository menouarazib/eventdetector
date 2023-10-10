import csv
import os
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from eventdetector_ts import OUTPUT_DIR, TimeUnit, MIDDLE_EVENT_LABEL
from eventdetector_ts.data.helpers_data import get_timedelta
from eventdetector_ts.plotter import logger, COLOR_TRUE, COLOR_PREDICTED, STYLE_PREDICTED, STYLE_TRUE, FIG_SIZE, PALETTE
from eventdetector_ts.plotter.helpers import event_to_rectangle


class Plotter:
    """
    The Plotter class is responsible for generating and saving plots of the predicted and true op, events, delta_t,...
        It provides a convenient way to visualize and compare the performance of a
        predictive model against the actual observed values.
    """

    def __init__(self, root_dir: str, time_unit: TimeUnit, width_events_s: float) -> None:
        """
        Initialize the Plotter object.

        Args:
            root_dir (str): The root directory for saving the plots.
            time_unit (TimeUnit): The unit time of the dataset.
            width_events_s (float): The width of events in the unit of time for the dataset.
        """

        self.val_losses = {}
        self.train_losses = {}
        self.val_loss_meta_model: list = []
        self.train_loss_meta_model: list = []
        self.width_events_s = width_events_s
        self.time_unit = time_unit
        # Whether to display the plots or not. Defaults to False.
        self.show = True
        self.root_dir = root_dir
        self.predicted_y: np.ndarray = np.empty(shape=(0,))
        self.test_y: np.ndarray = np.empty(shape=(0,))
        self.predicted_events: list = []
        self.true_events: pd.DataFrame = pd.DataFrame()
        self.delta_t: list = []
        self.working_dir = os.path.join(root_dir, OUTPUT_DIR)
        os.makedirs(self.working_dir)

    def set_show(self, show: bool) -> None:
        """
        Set show value
        Args:
            show (bool): Value to set for 'self.show'

        Returns:
            None
        """
        self.show = show

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

    def set_losses(self, train_losses: Dict[str, list], val_losses: Dict[str, list],
                   train_loss_meta_model: list, val_loss_meta_model: list) -> None:
        """
        Set losses of all trained models.
        Args:
            train_losses (Dict[str, list]): train losses.
            val_losses (Dict[str, list]): val losses.
            train_loss_meta_model (list): train loss for the metamodel.
            val_loss_meta_model (list): val loss for the metamodel.
        Returns:
            None
        """
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_loss_meta_model = train_loss_meta_model
        self.val_loss_meta_model = val_loss_meta_model

    def plot_prediction(self) -> None:
        """
        Plot the true and the predicted op and save it.

        Returns:
            None
        """

        logger.info("Plotting and saving the figure displaying the true and the predicted op")
        # Create the plot using Seaborn
        # Set the ggplot style
        sns.set(style="ticks", palette=PALETTE)
        plt.figure(figsize=FIG_SIZE)  # Set the figure size
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
        path = os.path.join(self.working_dir, "op.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()
        self.__save_op()

    def plot_predicted_events(self) -> None:
        """
        Plot the true and the predicted events and save it.

        Returns:
            None
        """

        logger.info("Plotting and saving the figure displaying the true events and the predicted events")
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.set(style="ticks", palette=PALETTE)

        for i, predicted_event in enumerate(self.predicted_events):
            rect1 = event_to_rectangle(event=predicted_event, width_events_s=self.width_events_s,
                                       time_unit=self.time_unit,
                                       color=COLOR_PREDICTED,
                                       style=STYLE_PREDICTED)
            ax.add_patch(rect1)

        for _, test_date in self.true_events[MIDDLE_EVENT_LABEL].items():
            rect1 = event_to_rectangle(event=test_date, width_events_s=self.width_events_s, time_unit=self.time_unit,
                                       color=COLOR_TRUE,
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
        path = os.path.join(self.working_dir, "events.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()
        self.__save_events()

    def plot_delta_t(self, bins=30) -> None:
        """
        Plots a histogram for delta t.

        Args:
            bins (int): The number of bins in the histogram. Default is 10.

        Returns:
              None
        """
        sns.set(style="ticks", palette=PALETTE)
        plt.figure(figsize=FIG_SIZE)

        sns.histplot(self.delta_t, bins=bins, binrange=(-self.width_events_s, self.width_events_s))

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

    def plot_losses(self):
        """
        Plot losses for all trained models.
        Returns:
            None
        """
        meta_model_was_used: bool = len(self.val_loss_meta_model) > 0

        sns.set(style="ticks", palette=PALETTE)
        if meta_model_was_used:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches((11, 8.5), forward=False)
        else:
            fig, ax1 = plt.subplots(figsize=FIG_SIZE)
        y_label = 'Loss'
        x_label = 'Epochs'
        colors = sns.color_palette(PALETTE, len(self.val_losses))
        lifestyle_val = '--'
        lifestyle_train = '-'
        for i, (model_name, val_loss) in enumerate(self.val_losses.items()):
            epochs = range(1, len(val_loss) + 1)
            train_loss = self.train_losses[model_name]
            ax1.plot(epochs, train_loss, linestyle=lifestyle_train, color=colors[i],
                     label='Training Loss - {}'.format(model_name))
            ax1.plot(epochs, val_loss, linestyle=lifestyle_val, color=colors[i],
                     label='Validation Loss - {}'.format(model_name))
            ax1.set_ylabel(y_label)
            ax1.set_xlabel(x_label)
            ax1.legend()

        if len(self.val_loss_meta_model) > 0:
            epochs_meta = range(1, len(self.val_loss_meta_model) + 1)
            ax2.plot(epochs_meta, self.train_loss_meta_model, linestyle=lifestyle_train, color='b',
                     label='Training Loss - Meta Model')
            ax2.plot(epochs_meta, self.val_loss_meta_model, linestyle=lifestyle_val, color='g',
                     label='Validation Loss - Meta Model')
            ax2.set_ylabel(y_label)
            ax2.set_xlabel(x_label)
            ax2.legend()

        fig.suptitle('Training and Validation Losses')
        plt.tight_layout()
        # Save the plot to a file
        path = os.path.join(self.working_dir, "losses.png")
        plt.savefig(path, dpi=300)
        # Show the plot
        if self.show:
            plt.show()

    def __save_events(self) -> None:
        """
        Save predicted events/true events to csv files.

        Returns:
            None
        """
        path = os.path.join(self.working_dir, "predicted_events.csv")
        radius = get_timedelta(float(self.width_events_s) / 2.0, self.time_unit)
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for i in range(len(self.predicted_events)):
                predicted_event = self.predicted_events[i]
                start_time = predicted_event - radius
                end_time = predicted_event + radius

                start_time = start_time.replace(microsecond=0)
                end_time = end_time.replace(microsecond=0)

                writer.writerow([start_time.isoformat(), end_time.isoformat()])

        path = os.path.join(self.working_dir, "true_events.csv")
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for _, test_date in enumerate(self.true_events[MIDDLE_EVENT_LABEL]):
                start_time = test_date - radius
                end_time = test_date + radius

                start_time = start_time.replace(microsecond=0)
                end_time = end_time.replace(microsecond=0)

                writer.writerow([start_time.isoformat(), end_time.isoformat()])

    def __save_op(self) -> None:
        """
        Save predicted/true Op into csv file.

        Returns:
            None
        """
        df = pd.DataFrame({'True-Op': self.test_y, 'Predicted-Op': self.predicted_y})
        path = os.path.join(self.working_dir, "op.csv")
        df.to_csv(path, index=True, sep=" ")

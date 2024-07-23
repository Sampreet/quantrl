#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with plotters."""

__name__    = 'quantrl.plotters'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-12-08"
__updated__ = "2024-07-23"

# dependencies
from io import BytesIO
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# OpenMP configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# matplotlib configuration
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

class TrajectoryPlotter(object):
    """Plotter for trajectories.

    Initializes ``axes_rows``, ``fig``, ``axes`` and ``lines``.

    Parameters
    ----------
    axes_args: list
        Lists of axis properties. The first element of each entry is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
    axes_lines_max: int, default=10
        Maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``10``.
    axes_cols: int, default=2
        Number of columns in the figure. Default is ``2``.
    show_title: bool, default=True
        Option to display the trajectory index as title.
    save_dir: str, default=None
        Directory to save the plots on each update. If ``None``, the plots are not saved.
    """

    def __init__(self,
        axes_args:list,
        axes_lines_max:int=10,
        axes_cols:int=2,
        show_title:bool=True,
        save_dir:str=None
    ):
        """Class constructor for TrajectoryPlotter."""

        # validate
        assert axes_lines_max >= 0, "parameter ``axes_lines_max`` should be a non-negative integer"
        assert axes_cols > 0, "parameter ``axes_cols`` should be a positive integer"

        # set attributes
        self.axes_args = axes_args
        self.axes_lines_max = axes_lines_max
        self.axes_cols = axes_cols
        self.show_title = show_title
        self.save_dir = save_dir

        # create directories if requires saving
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
            except OSError:
                pass

        # turn on interactive mode
        plt.ion()

        # initialize variables
        self.axes_rows = int(np.ceil(len(self.axes_args) / self.axes_cols))
        self.fig = plt.figure(figsize=(6.0 * self.axes_cols, 3.0 * self.axes_rows))
        self.gspec = GridSpec(self.axes_rows, self.axes_cols, figure=self.fig, width_ratios=[0.2] * self.axes_cols)
        self.axes = list()
        self.lines = None

        # format frame
        for i in range(self.axes_rows):
            for j in range(self.axes_cols):
                if i * self.axes_cols + j >= len(self.axes_args):
                    break
                # new subplot
                ax = self.fig.add_subplot(self.gspec[i, j])
                ax_args = self.axes_args[i * self.axes_cols + j]
                # y-axis label, limits and scale
                ax.set_xlabel(ax_args[0])
                ax.xaxis.set_ticks_position('both')
                ax.set_ylabel(ax_args[1])
                ax.set_ylim(ymin=ax_args[2][0], ymax=ax_args[2][1])
                ax.yaxis.set_ticks_position('both')
                ax.set_yscale(ax_args[3])
                self.axes.append(ax)
        if self.show_title:
            self.fig.suptitle('#0')
        self.fig.tight_layout()

        # initialize buffers
        self.frames = list()

    def plot_lines(self,
        xs,
        Y,
        traj_idx=0,
        update_buffer=False
    ):
        """Method to plot new lines.

        Parameters
        ----------
        xs: :class:`numpy.ndarray` or list
            Common X-axis values.
        Y: :class:`numpy.ndarray` or list
            Trajectory data points with shape ``(t_dim, n_data)``.
        traj_idx: int, default=0
            Index of the current trajectory.
        update_buffer: bool, default=False
            Option to update the frame buffer.
        """

        # dim existing lines
        if self.lines is not None:
            for line in self.lines:
                line.set_alpha(0.1)

        # add new lines
        self.lines = list()
        for i, ax in enumerate(self.axes):
            if self.axes_lines_max and len(ax.get_lines()) >= self.axes_lines_max:
                line = ax.get_lines()[0]
                line.remove()
            self.lines.append(ax.plot(xs, Y[:, i])[0])
        if self.show_title and self.fig._suptitle is not None:
            self.fig.suptitle('#' + str(traj_idx))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # update frames
        if update_buffer:
            frame_buffer = BytesIO()
            plt.savefig(frame_buffer, format='png')
            self.frames.append(Image.open(frame_buffer))

        # save plot
        if self.save_dir is not None:
            self.save_plot(
                file_name=self.save_dir + '/traj_' + str(traj_idx)
            )
            self.save_plot(
                file_name=self.save_dir + '_latest'
            )

    def make_gif(self,
        file_name:str
    ):
        """Method to create a gif file from the frame buffer.

        Parameters
        ----------
        file_name: str
            Name of the file without its extension.
        """

        # if no frames in buffer
        if len(self.frames) == 0:
            return

        # dump buffer
        frame = self.frames[0]
        frame.save(file_name + '.gif', format='GIF', append_images=self.frames[1:], save_all=True, duration=50, loop=0)

        # reset buffer
        del self.frames
        self.frames = list()

    def save_plot(self,
        file_name:str
    ):
        """Method to save the plot.

        Parameters
        ----------
        file_name: str
            Name of the file without its extension.
        """

        plt.savefig(file_name)

    def hold_plot(self):
        """Method to hold the plot."""

        # turn off interactive mode
        plt.ioff()
        # show plot
        plt.show()

    def close(self):
        """Method to close the plotter."""

        # reset buffer
        del self.frames
        self.frames = None

        # close plotter
        plt.close()

        # clean
        del self

class LearningCurvePlotter(object):
    """Plotter for learning curve.

    Initializes ``fig``, ``ax`` and ``data``.

    Parameters
    ----------
    axes_args: list
        Lists of axis properties. The first element of each entry is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
    average_over: int, default=100
        Number of points to average over.
    percentiles: list, default=[25, 50, 75]
        Percentile values for intraquartile ranges.
    """

    def __init__(self,
        axis_args:list,
        average_over:int=100,
        percentiles:list=[25, 50, 75]
    ):
        """Class constructor for LearningCurvePlotter."""

        # set attributes
        self.axis_args = axis_args
        self.average_over = average_over
        self.percentiles = percentiles

        # turn on interactive mode
        plt.ion()

        # initialize plot
        self.fig = plt.figure(figsize=(6.0, 3.0))
        self.ax = plt.gca()
        self.ax.set_xlabel(self.axis_args[0])
        self.ax.set_ylabel(self.axis_args[1])
        self.ax.set_ylim(ymin=self.axis_args[2][0], ymax=self.axis_args[2][1])
        self.ax.set_yscale(self.axis_args[3])
        self.fig.tight_layout()

        # initialze buffer
        self.data = list()
        self.line = None
        self.line_faint = None

    def add_data(self,
        data_rewards:np.ndarray,
        renew:bool=False,
        color:str='k',
        style:str='-',
        width:float=1.5
    ):
        """Method to add reward data.

        Parameters
        ----------
        data_rewards: :class:`numpy.ndarray`
            Reward data for the plot.
        renew: bool, default=False
            Option to create a new list
        color: str, default='k'
            Line color of the plot.
        style: str, default='-'
            Line style of the plot.
        width: float, default=1.5
            Line width of the plot.
        """

        # if averaging opted
        if self.average_over is not None:
            data_rewards_smooth = np.convolve(data_rewards, np.ones((self.average_over, )) / float(self.average_over), mode='valid')

        # update data
        if renew:
            self.data = list()
            self.line = None
            self.line_faint = None
        self.data.append(data_rewards_smooth)
        if self.line_faint is not None:
            self.line_faint.remove()
        if self.line is not None:
            self.line.remove()
        xs = list(range(self.average_over, len(self.data[0]) + self.average_over))

        # if single entry
        if len(self.data) == 1:
            q_mean = data_rewards_smooth
            self.line_faint = self.ax.plot(xs, data_rewards[self.average_over - 1:], c=color, alpha=0.1, linewidth=0.5)[0]
        else:
            # if interquartile ranges given
            if self.percentiles is not None:
                q_min, q_mean, q_max = np.percentile(self.data, self.percentiles, 0)
                self.line_faint = self.ax.fill_between(xs, q_min, q_max, facecolor=color, alpha=0.1)
            # calculate mean
            else:
                q_mean = np.mean(self.data, 0)
                self.line_faint = self.ax.plot(xs, q_mean, c=color, alpha=0.1, linewidth=0.5)[0]

        # add new lines
        self.line = self.ax.plot(xs, q_mean, c=color, linestyle=style, linewidth=width)[0]

    def save_plot(self,
        file_name:str
    ):
        """Method to save the plot.

        Parameters
        ----------
        file_name: str
            Name of the file without its extension.
        """

        plt.savefig(file_name)

    def hold_plot(self):
        """Method to hold the plot."""

        # turn off interactive mode
        plt.ioff()
        # show plot
        plt.show()

    def close(self):
        """Method to close the plotter."""

        # close plotter
        plt.close()

        # clean
        del self
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with plotters."""

__name__    = 'quantrl.plotters'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-12-08"
__updated__ = "2024-02-17"

# dependencies
from io import BytesIO
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
plt.rcParams['figure.subplot.hspace'] = 0.2
plt.rcParams['figure.subplot.wspace'] = 0.25

class TrajectoryPlotter(object):
    """Plotter for trajectories.

    Initializes ``axes_rows``, ``fig``, ``axes`` and ``lines``.

    Parameters
    ----------
    axes_args: list, optional
        Lists of axis properties. The first element of each entry is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[]``.
    axes_lines_max: int, optional
        Maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
    axes_cols: int, optional
        Number of columns in the figure. Default is ``3``.
    show_title: bool, optional
        Option to display the trajectory index as title. Default is ``True``.
    """

    def __init__(self,
        axes_args:list=[],
        axes_lines_max:int=100,
        axes_cols:int=3,
        show_title:bool=True
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

        # turn on interactive mode
        plt.ion()

        # initialize variables
        self.axes_rows = int(np.ceil(len(self.axes_args) / self.axes_cols))
        self.fig = plt.figure(figsize=(6.0 * self.axes_cols, 3.0 * self.axes_rows))
        self.axes = list()
        self.lines = None

        # format frame
        for i in range(self.axes_rows):
            for j in range(self.axes_cols):
                if i * self.axes_cols + j >= len(self.axes_args):
                    break
                # new subplot
                ax = self.fig.add_subplot(int(self.axes_rows * 100 + self.axes_cols * 10 + i * self.axes_cols + j + 1))
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
        xs: list
            Common X-axis values.
        Y: list
            Trajectory data points with shape ``(t_dim, n_data)``.
        traj_idx: int
            Index of the current trajectory.
        update_buffer: bool
            Option to update the frame buffer
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

    def update_lines(self,
        y_js=[1.0],
        j=0
    ):
        """Method to update lines.

        Parameters
        ----------
        y_js: list
            New values of trajectory data with shape ``(action_interval, n_data)``.
        j: int
            Index of the current update.
        """

        # update existing lines
        for i in range(len(self.lines)):
            ys = self.lines[i].get_ydata()
            ys[j - len(y_js) + 1:j + 1] = y_js
            self.lines[i].set_ydata(ys)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
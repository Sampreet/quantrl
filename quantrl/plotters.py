#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module with plotters."""

__name__    = 'quantrl.plotters'
__authors__ = ['Sampreet Kalita']
__created__ = '2023-12-08'
__updated__ = '2024-01-16'

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class BaseTrajectoryPlotter(object):
    """Base plotter for trajectories.

    Initializes ``axes_rows``, ``fig``, ``axes`` and ``lines``.

    Parameters
    ----------
    axes_args: list, optional
        Lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
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
        """Class constructor for BasePlotter."""

        # validate
        assert axes_lines_max >= 0, 'parameter ``axes_lines_max`` should be a non-negative integer'
        assert axes_cols > 0, 'parameter ``axes_cols`` should be a positive integer'

        # set attributes
        self.axes_args = axes_args
        self.axes_lines_max = axes_lines_max
        self.axes_cols = axes_cols
        self.show_title = show_title

        # matplotlib configuration
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['figure.subplot.hspace'] = 0.2
        plt.rcParams['figure.subplot.wspace'] = 0.25
        plt.ion()

        # initialize
        self.axes_rows = int(np.ceil(len(self.axes_args) / self.axes_cols))
        self.fig = plt.figure(figsize=(6.0 * self.axes_cols, 3.0 * self.axes_rows))
        self.axes = list()
        self.lines = None
        for i in range(self.axes_rows):
            for j in range(self.axes_cols):
                if i * self.axes_cols + j >= len(self.axes_args):
                    break
                # new subplot
                ax = self.fig.add_subplot(int(self.axes_rows * 100 + self.axes_cols * 10 + i * self.axes_cols + j + 1))
                ax_args = self.axes_args[i * self.axes_cols + j]
                # y-axis label, limits and scale
                ax.set_xlabel(ax_args[0])
                ax.set_ylabel(ax_args[1])
                ax.set_ylim(ymin=ax_args[2][0], ymax=ax_args[2][1])
                ax.set_yscale(ax_args[3])
                self.axes.append(ax)
        if self.show_title:
            self.fig.suptitle('#0')

    def plot_lines(self,
        xs,
        Y,
        idxs,
        traj_idx=0
    ):
        """Method to plot new lines.
        
        Parameters
        ----------
        xs: list
            Common X-axis values.
        Y: list
            Trajectory data points with shape ``(t_dim, n_data)``.
        idxs: list
            Indices of data points to plot from ``n_data``.
        traj_idx: int
            Index of the current trajectory.
        """

        # dim existing lines
        if self.lines is not None:
            for line in self.lines:
                line.set_alpha(0.1)

        # add new lines
        self.lines = list()
        for i, ax in enumerate(self.axes):
            ys = Y[:, idxs[i]]
            if self.axes_lines_max and len(ax.get_lines()) >= self.axes_lines_max:
                line = ax.get_lines()[0]
                line.remove()
            self.lines.append(ax.plot(xs, ys)[0])
        if self.show_title and self.fig._suptitle is not None:
            self.fig.suptitle('#' + str(traj_idx))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_plot(self):
        """Method to hold the plot."""

        plt.ioff()
        plt.show()

    def update_lines(self,
        y_js=[1.0],
        j=0,
        idxs=[0]
    ):
        """Method to update lines.
        
        Parameters
        ----------
        y_js: list
            New values of trajectory data with shape ``(action_interval, n_data)``.
        j: int
            Index of the current update.
        idxs: list
            Indices of data points to plot from ``n_data``.
        """

        # update existing lines
        for i in range(len(self.lines)):
            ys = self.lines[i].get_ydata()
            ys[j - len(y_js) + 1:j + 1] = y_js[:, idxs[i]]
            self.lines[i].set_ydata(ys)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
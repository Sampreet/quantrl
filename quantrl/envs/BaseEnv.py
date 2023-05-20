#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module to interface quantum systems with gym environments."""

__name__    = 'quantrl.envs.BaseEnv'
__authors__ = ['Sampreet Kalita']
__created__ = '2023-04-25'
__updated__ = '2023-05-20'

# dependencies
import gym
import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.integrate as si

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# base environment
class BaseEnv(gym.Env):
    r"""Base environment to interface quantum systems.

    Initializes ``action_space``, ``observation_space``, ``file_name``, ``all_data``, ``t_dim``, ``t_norms`` and ``plotter``.

    Parameters
    ----------
    n_trajectories: int
        Number of trajectories.
    t_norm_max: float
        Maximum time for each trajectory in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    action_max: list
        Maximum value of action. Can be negative or non-negative depending on ``action_space_range``.
    action_interval: int, optional
        Number of steps after which the action updates. Must be positive. Default is `1`.
    reward_max: float, optional
        Maximum value of reward (implemented in children). Default is ``1.0``.
    reward_noise: float, optional
        Noise in the reward function (implemented in children). Default is ``0.0``.
    action_space_range: list, optional
        Range of the actions obtained from the network. Default is ``[-1, 1]``.
    observation_space_range: list, optional
        Range of the observations. Default is ``[-1e6, 1e6]``.
    observation_space_shape: tuple, optional
        Shape of the observations. Default is ``(10, )``.
    save_properties: bool, optional
        Option to save additional properties for each time step. Requires ``get_properties()`` method implemented in children.
    solver_type: str, optional
        Solver to evolve each time step. Options are ``"qtraj"`` for quantum trajectories using the ``get_H_eff(t)`` method implemented in children (see Note), ``"wiener"`` for Wiener process using the ``get_M(t)`` method implemented in children and ``"scipy"`` for ``:class:scipy.integrate`` using the default ``func_ode(t, y, c)`` method implemented in children. Default is ``"scipy"``.
    plot: bool, optional
        Option to plot the trajectories using ``:class:BaseEnvPlotter``. Default is ``True``.
    plot_interval: int, optional
        Interval at which the trajectories are updated. Must be non-negative. ``0`` means update after the trajectory is complete. Default is ``0``.
    plot_idxs: list, optional
        Positions of the data array to plot at each time step. Default is ``[-1]``.
    axes_args: list, optional
        Lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
    axes_lines_max: int, optional
        Maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
    axes_cols: int, optional
        Number of columns in the figure. Default is ``3``.
    dir_prefix: str, optional
        Prefix of the directory where the data will be stored. Default is ``"data"``.

    .. note:: ``observation_space_shape`` and ``solver_type`` may be different for different systems.
    """
    def __init__(self,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        action_max:list,
        action_interval:int=1,
        reward_max:float=1.0,
        reward_noise:float=0.0,
        action_space_range:list=[-1, 1],
        observation_space_range:list=[-1e6, 1e6],
        observation_space_shape:tuple=(10, ),
        save_properties:bool=False,
        solver_type:str='scipy',
        plot:bool=True,
        plot_interval:int=0,
        plot_idxs:list=[-1],
        axes_args:list=[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']],
        axes_lines_max:int=100,
        axes_cols:int=3,
        dir_prefix:str='data'
    ):
        # validate
        assert t_norm_max > t_norm_ssz, 'maximum normalized time should be greater than the normalized step size'
        assert action_interval > 0, 'parameter ``action_interval`` should be a positive integer'
        assert plot_interval >= 0, 'parameter ``plot_interval`` should be a non-negative integer'
        assert len(plot_idxs) == len(axes_args), 'number of indices for plot should match number of axes arguments'

        # gym environment
        super().__init__()
        self.action_space       = spaces.Box(
            low=action_space_range[0],
            high=action_space_range[1],
            shape=np.shape(action_max),
            dtype=np.float32
        )
        self.observation_space  = spaces.Box(
            low=observation_space_range[0],
            high=observation_space_range[1],
            shape=observation_space_shape,
            dtype=np.float32
        )

        # trajectory variables
        self.n_trajectories = n_trajectories
        self.t_norm_max     = np.float32(t_norm_max)
        self.t_norm_ssz     = np.float32(t_norm_ssz)
        self.action_max     = np.array(action_max, dtype=np.float32)
        self.action_interval= action_interval
        self.reward_max     = np.float32(reward_max)
        self.reward_noise   = np.float32(reward_noise)
        self.save_properties= save_properties
        self.solver_type    = solver_type

        # solver variables
        self.traj_idx       = -1
        self.t_dim          = int(t_norm_max / t_norm_ssz) + 1
        self.t_norms        = np.linspace(0.0, t_norm_max, self.t_dim)
        if 'qtraj' in self.solver_type:
            self.epsis  = np.random.random_sample((n_trajectories, self.t_dim, 2))
        elif 'wiener' in self.solver_type:
            self.W_is   = np.sqrt(self.t_norm_ssz) * np.random.random_sample((n_trajectories, self.t_dim))

        # initialize IO
        self.io             = BaseEnvIO(
            file_name_prefix=dir_prefix + '_' + '_'.join([
                str(n_trajectories),
                str(t_norm_max),
                str(t_norm_ssz),
                str(action_max),
                str(action_interval),
                str(reward_max),
                str(reward_noise)]
            ) + '/env'
        )

        # plot options
        self.plot           = plot
        self.plot_interval  = int(plot_interval)
        self.plot_idxs      = plot_idxs
        self.axes_args      = axes_args
        self.axes_lines_max = axes_lines_max
        self.axes_cols      = axes_cols

        # initialize plotter
        if plot:
            self.plotter    = BaseEnvPlotter(
                axes_args=axes_args,
                axes_lines_max=axes_lines_max,
                axes_cols=axes_cols,
                show_title=True
            )

    def reset(self):
        """Method to reset all variables for a new trajectory.
        
        Returns
        -------
        observations: :class:numpy.ndarray
            Inititial array of observations.
        """
        # reset variables
        self.traj_idx   += 1
        self.t_idx      = 0
        self.t          = np.float32(0.0)
        self.t_next     = self.ts[self.t_idx + self.action_interval]
        self.R          = 0.0
        self.action     = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.observation= np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # reset data
        data_0          = self.get_data(reset=True)
        self.data       = np.zeros((self.t_dim, np.shape(data_0)[0]), dtype=np.float32)
        self.data[0]    = self.get_data(reset=True)

        # reset plots
        if self.plot and self.plot_interval:
            self.plotter.plot_lines(xs=self.t_norms[::self.plot_interval], Y=self.data, idxs=self.plot_idxs)

        return self.observation
    
    def get_data(self, reset=False):
        """Method to reset all variables for a new trajectory.

        Parameters
        ----------
        reset: bool, optional
            Option to reset the system. Default is ``False``.

        Returns
        -------
        data: :class:numpy.ndarray
            Array of floats.
            The first element contains the current time.
            The next ``n_actions`` elements contain the actions.
            The next ``n_observations`` elements contain the observations.
            The next ``n_properties`` elements contain the properties.
            The final element is the current cummulative reward.
        """
        if reset:
            self.reset_env()
        properties = list()
        if self.save_properties:
            properties = self.get_properties()
        return np.array([self.t] + [a for a in self.action] + [o for o in self.observation] + properties + [self.R], dtype=np.float32)

    def step(self, action):
        """Method to reset all variables for a new trajectory.

        Parameters
        ----------
        action: :class:numpy.ndarray
            Array of actions at current time.
        
        Returns
        -------
        observations: :class:numpy.ndarray
            Array of observations at current time.
        reward: :class:numpy.float32
            Reward calculated for the action interval.
        done: bool
            Flag to terminate trajectory.
        info: dict
            Additional information.
        """
        self.action = action * self.action_max
        
        # quantum trajectories
        if self.solver_type == 'qtraj':
            observations = self._step_qtraj()
        # Wiener processes
        elif self.solver_type == 'wiener':
            observations = self._step_wiener()
        # coupled ODEs
        else:
            observations = self._step_scipy()

        # update data
        prev_reward = self.R
        for i, observation in enumerate(observations):
            self.t_idx                  += 1
            self.t                      = self.ts[self.t_idx]
            self.observation            = observation
            self.R                      += self.get_reward() if i == self.action_interval - 1 else 0
            self.data[self.t_idx]       = self.get_data()
            # update plots
            if self.plot and self.plot_interval and self.t_idx % self.plot_interval == 0:
                self.plotter.update_lines(y_js=self.get_data(), j=int(self.t_idx / self.plot_interval), idxs=self.plot_idxs)

        # check if completed
        done = False if self.t_idx + self.action_interval < self.t_dim else True
        if done:
            if self.plot and not self.plot_interval:
                self.plotter.plot_lines(xs=self.t_norms, Y=self.data, idxs=self.plot_idxs)
            self.io.update_all_data(self.data)
        else:
            self.t_next = self.ts[self.t_idx + self.action_interval]

        return np.float32(self.observation), float(self.R - prev_reward), done, {}
    
    def _step_qtraj(self):
        """Method to implement one step of a quantum trajectory."""
        raise NotImplementedError
    
    def _step_wiener(self):
        """Method to implement one step of a Wiener process."""
        self.t_eval = np.linspace(self.t, self.t_next, self.action_interval + 1, dtype=np.float32)

        # TODO: vectorize
        observations = np.zeros((self.action_interval + 1, self.observation_space.shape[0]), dtype=np.float32)
        observations = np.zeros((self.action_interval + 1, self.observation_space.shape[0]), dtype=np.float32)
        observations[0] = self.observation
        for i in range(1, self.action_interval + 1):
            M_i = self.get_M_i(self.t_idx + i, self.action)
            n_i = self.get_noise_prefix() * self.W_is[self.traj_idx, self.t_idx + i]
            observations[i] = M_i.dot(observations[i - 1]) + n_i

        return observations[1:]

    def _step_scipy(self):
        """Method to implement one step of an IVP solver."""
        self.t_eval = np.linspace(self.t, self.t_next, self.action_interval + 1, dtype=np.float32)

        # integrate
        sol = si.solve_ivp(
            fun=self.func_ode,
            t_span=[self.t, self.t_next],
            method='RK45',
            y0=self.observation,
            t_eval=self.t_eval,
            args=(self.action, )
        )

        return np.transpose(sol.y)[1:]

    def save_all_data(self):
        """Method to save all data."""
        self.io.save_all_data()
    
    def replay_all_data(self, offset=0):
        """Method to replay all data."""
        all_data = self.io.load_all_data()
        for Y in all_data[offset:]:
            self.plotter.plot_lines(xs=self.t_norms, Y=Y, idxs=self.plot_idxs)
        self.plotter.show_plot()

    def plot_learning_curve(self, axis_args=['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']):
        all_data = self.io.load_all_data()
        plotter = BaseEnvPlotter(
            axes_args=[axis_args],
            axes_lines_max=1,
            axes_cols=1,
            show_title=False
        )
        plotter.plot_lines(xs=list(range(len(all_data))), Y=np.squeeze(all_data[:, -1, :]), idxs=[-1])
        plotter.show_plot()

class BaseEnvIO(object):
    def __init__(self,
        file_name_prefix:str
    ):
        # io variables
        self.file_name_prefix = file_name_prefix
        try:
            os.makedirs(self.file_name_prefix[:self.file_name_prefix.rfind('/')], exist_ok=True)
        except OSError:
            pass

        # initialize data
        self.all_data = list()

    def update_all_data(self, data):
        self.all_data.append(data)

    def save_all_data(self):
        np.savez_compressed(self.file_name_prefix + '.trajectories.npz', np.array(self.all_data))

    def load_all_data(self):
        return np.load(self.file_name_prefix + '.trajectories.npz')['arr_0']

class BaseEnvPlotter(object):
    def __init__(self,
        axes_args:list=[],
        axes_lines_max:int=100,
        axes_cols:int=3,
        show_title:bool=True
    ):
        # validate
        assert axes_lines_max >= 0, 'parameter ``axes_lines_max`` should be a non-negative integer'
        assert axes_cols > 0, 'parameter ``axes_cols`` should be a positive integer'

        # set attributes
        self.axes_args      = axes_args
        self.axes_lines_max = axes_lines_max
        self.axes_cols      = axes_cols
        self.show_title     = show_title

        # matplotlib configuration
        plt.rcParams['font.family']             = 'Times New Roman'
        plt.rcParams['font.size']               = 16
        plt.rcParams['mathtext.fontset']        = 'cm'
        plt.rcParams['figure.subplot.hspace']   = 0.2
        plt.rcParams['figure.subplot.wspace']   = 0.25
        plt.ion()

        # initialize
        self.axes_rows  = int(np.ceil(len(self.axes_args) / self.axes_cols))
        self.fig        = plt.figure(figsize=(6.0 * self.axes_cols, 3.0 * self.axes_rows))
        self.axes       = list()
        self.lines      = None
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

    def plot_lines(self, xs=list(range(11)), Y=[[i] for i in range(11)], idxs=[0]):
        if self.lines is not None:
            for line in self.lines:
                line.set_alpha(0.1)
        self.lines = list()
        for i, ax in enumerate(self.axes):
            ys = Y[:, idxs[i]]
            if self.axes_lines_max and len(ax.get_lines()) >= self.axes_lines_max:
                line = ax.get_lines()[0]
                line.remove()
            self.lines.append(ax.plot(xs, ys)[0])
        if self.show_title and self.fig._suptitle is not None:
            self.fig.suptitle('#' + str(int(self.fig._suptitle.get_text()[1:]) + 1))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_plot(self):
        plt.ioff()
        plt.show()

    def update_lines(self, y_js=[1.0], j=0, idxs=[0]):
        for i in range(len(self.lines)):
            ys = self.lines[i].get_ydata()
            ys[j] = y_js[idxs[i]]
            self.lines[i].set_ydata(ys)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module with base environments for reinforcement learning."""

__name__    = 'quantrl.envs.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-02-10"

# dependencies
from tqdm import tqdm
import gymnasium
import numpy as np

# quantrl modules
from ..io import FileIO
from ..plotters import TrajectoryPlotter

# TODO: Interface ConsoleIO
# TODO: Implement kwargs

class BaseGymEnv(gymnasium.Env):
    r"""Gymnasium-based base environment for reinforcement-learning.

    Initializes ``action_space``, ``observation_space``, ``t_norms``, ``io`` and ``plotter``.

    The interfaced environment needs to implement ``_step``, ``reset_Observations``, ``get_Properties`` and ``get_Reward`` methods.
    Refer to **Notes** below for their implementations.

    Parameters
    ----------
    n_trajectories: int
        Total number of trajectories.
    t_norm_max: float
        Maximum time for each trajectory in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    action_maxs: list
        Maximum values of each action.
    actions_interval: int, optional
        Interval at which the actions are updated. Must be positive. Default is ``1``.
    reward_max: float, optional
        Maximum value of reward (implemented in children). Default is ``1.0``.
    reward_noise: float, optional
        Noise in the reward function (implemented in children). Default is ``0.0``.
    action_space_range: list, optional
        Range of the actions obtained from the network. The output is scaled by the corresponding action multiplier. Default is ``[-1, 1]``.
    observation_space_range: list, optional
        Range of the observations. Default is ``[-1e6, 1e6]``.
    observation_space_shape: tuple, optional
        Shape of the observations. Default is ``(10, )``.
    save_properties: bool, optional
        Option to save additional properties for each time step. Requires ``get_properties()`` method implemented in children.
    plot: bool, optional
        Option to plot the trajectories using ``:class:BaseTrajectoryPlotter``. Default is ``True``.
    plot_interval: int, optional
        Number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
    plot_idxs: list, optional
        Indices of the data values required to plot at each time step. Default is ``[-1]`` for the cummulative reward.
    axes_args: list, optional
        Lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
    axes_lines_max: int, optional
        Maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
    axes_cols: int, optional
        Number of columns in the figure. Default is ``3``.
    dir_prefix: str, optional
        Prefix of the directory where the data will be stored. Default is ``'data'``.
    max_trajectories_per_file: int, optional
        Maximum number of trajectory data to save per file. Default is ``100``.

    .. note:: ``observation_space_shape`` and ``solver_type`` may be different for different systems.

    Notes
    -----
        The following required methods follow a strict formatting:
            ====================    ================================================
            method                  returns
            ====================    ================================================
            _step                   the updated observations with shape ``(action_interval + 1, n_observations)``, formatted as ``_step(actions)``, where ``actions`` is the array of actions with shape ``(n_actions, )`` multiplied by ``action_maxs``.
            get_Properties          the properties calculated from ``Observations`` with shape ``(action_interval + 1, n_properties)``.
            get_Reward              the reward calculated using the observations or the properties with shape ``(action_interval, )``. The class attributes ``reward_max`` and ``reward_noise`` can be utilized here.
            reset_Observations      ``None``. This method is used to reset the first entry (index ``0``) of the ``Observations`` variable which has shape ``(action_interval + 1, n_observations)`` with the initial values of the observations.
            ====================    ================================================
    """

    axis_args_learning_curve = ['Episodes $N$', 'Cummulative Reward $\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']
    """list: Axis arguments to plot the learning curve."""

    def __init__(self,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        action_maxs:list,
        action_interval:int=1,
        has_delay:bool=False,
        reward_max:float=0.1,
        reward_noise:float=0.0,
        action_space_range:list=[-1, 1],
        observation_space_range:list=[-1e6, 1e6],
        observation_space_shape:tuple=(10, ),
        save_properties:bool=False,
        plot:bool=True,
        plot_interval:int=1,
        plot_idxs:list=[-1],
        axes_args:list=[
            ['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']
        ],
        axes_lines_max:int=100,
        axes_cols:int=3,
        dir_prefix:str='data',
        max_trajectories_per_file:int=100
    ):
        """Class constructor for BaseGymEnv."""

        # validate
        assert t_norm_max > t_norm_ssz, 'maximum normalized time should be greater than the normalized step size'
        assert action_interval > 0, 'parameter ``action_interval`` should be a positive integer'
        assert plot_interval >= 0, 'parameter ``plot_interval`` should be a non-negative integer'
        assert plot_interval < n_trajectories if plot else True, 'parameter ``plot_interval`` should be a less than parameter ``n_trajectories``'
        assert len(plot_idxs) == len(axes_args), 'number of indices for plot should match number of axes arguments'
        
        # trajectory constants
        self.n_trajectories = n_trajectories
        # time constants
        self.t_norm_max = t_norm_max
        self.t_norm_ssz = t_norm_ssz
        # truncate before maximum time if not divisible
        self.t_dim = int(self.t_norm_max / self.t_norm_ssz) + 1
        self.t_norm_mul = t_norm_mul
        self.T_norm = np.arange(self.t_dim, dtype=np.float_) * self.t_norm_ssz
        self.T = self.T_norm * t_norm_mul
        # step constants
        self.action_maxs = np.array(action_maxs, dtype=np.float_)
        self.action_interval = action_interval
        # extend one step if not divisible
        _action_dim = (self.t_dim - 1) / self.action_interval
        self.total_timesteps = self.n_trajectories * (int(_action_dim + 1) if _action_dim - int(_action_dim) > 0 else int(_action_dim))
        # align delay with action interval
        self.has_delay = has_delay
        self.t_delay = self.T[self.action_interval] - self.T[0]
        # reward constants
        self.reward_max = np.float_(reward_max)
        self.reward_noise = np.float_(reward_noise)
        # data constants
        self.observation_space_range = observation_space_range
        self.save_properties = save_properties
        self.file_prefix = dir_prefix + '_' + '_'.join([
            str(n_trajectories),
            str(t_norm_max),
            str(t_norm_ssz),
            str(action_maxs),
            str(action_interval),
            str(reward_max),
            str(reward_noise)]
        ) + '/env'
        # plot constants
        self.plot = plot
        self.plot_interval = plot_interval
        self.plot_idxs = plot_idxs
        
        # initialize environment
        super().__init__()
        self.action_space = gymnasium.spaces.Box(
            low=action_space_range[0],
            high=action_space_range[1],
            shape=self.action_maxs.shape,
            dtype=np.float_
        )
        self.observation_space = gymnasium.spaces.Box(
            low=observation_space_range[0],
            high=observation_space_range[1],
            shape=observation_space_shape,
            dtype=np.float_
        )
        self.n_actions = self.action_space.shape[0]
        self.n_observations = self.observation_space.shape[0]

        # initialize IO
        self.io = FileIO(
            disk_cache_dir=self.file_prefix + '_cache',
            max_cache_size=max_trajectories_per_file
        )

        # initialize plotter
        if self.plot:
            self.plotter = TrajectoryPlotter(
                axes_args=axes_args,
                axes_lines_max=axes_lines_max,
                axes_cols=axes_cols,
                show_title=True
            )

        # initialize buffers
        self.traj_idx = -1
        self.actions = np.zeros(self.n_actions, dtype=np.float_)
        self.Observations = np.zeros((self.action_interval + 1, self.n_observations), dtype=np.float_)
        self.Rs = np.zeros((self.n_trajectories, 1), dtype=np.float_)

        # validate environment
        try:
            self.reset_Observations()
            self.Observations[1:] = self.Observations[0]
            self.n_data_elements = self.n_actions + self.n_observations + 2
            if self.save_properties:
                n_properties = np.shape(self.get_Properties())[1]
                self.n_data_elements += n_properties
            self.R = self.get_Reward()[-1]
        except AttributeError as error:
            print(f"Missing required method ({error}). Refer to **Notes** of ``:class:quantrl.envs.base.BaseGymEnv`` for its implementation.")
            exit()

    def reset(self,
        seed:float=None,
        options:dict=None
    ):
        """Method to reset all variables for a new trajectory.

        Parameters
        ----------
        seed: float
            Seed value for the reset.
        options: dict
            Options for the reset.
        
        Returns
        -------
        observations: :class:`numpy.ndarray`
            Inititial array of observations.
        info: str
            Information on the reset.
        """

        # update trajectory count
        self.traj_idx += 1
        # reset time
        self.t_idx = 0
        self.t = np.float_(0.0)
        # reset reward
        self.R = 0.0
        # reset observations
        self.reset_Observations()
        self.Observations[1:] = self.Observations[0]
        # reset data
        self.data = np.zeros((self.t_dim, self.n_data_elements), dtype=np.float_)  
        # reset plots
        if self.plot and not self.plot_interval:
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=self.data[:, self.plot_idxs],
                traj_idx=self.traj_idx
            )

        return self.Observations[0], None
    
    def render(self):
        """Method to render one step."""

        pass

    def step(self,
        actions
    ):
        """Method to take one single step.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Array of actions at current time.
        
        Returns
        -------
        observations: :class:`numpy.ndarray`
            Array of observations at current time.
        reward: float
            Reward calculated for the action interval.
        terminated: bool
            Flag to terminate trajectory.
        truncated: bool
            Flag to truncate trajectory.
        info: dict
            Additional information.
        """

        # update actions
        self.actions[:] = actions * self.action_maxs
        # store previous reward
        prev_reward = self.R

        # set evaluation times
        self.T_step = self.T[self.t_idx:self.t_idx + self.action_interval + 1] if self.t_idx + self.action_interval < self.t_dim else self.T[self.t_idx:]

        # step and update data
        self.Observations = self._step(
            actions=self.actions
        )
        self._update()

        # check if out of bounds
        truncated = np.max(self.Observations) > self.observation_space_range[1] or np.min(self.Observations) < self.observation_space_range[0]

        # check if completed
        terminated = False if self.t_idx + 1 < self.t_dim else True
        if terminated or truncated:
            print(f'Trajectory #{self.traj_idx} truncated') if truncated else True
            # update plot
            if self.plot and self.plot_interval and self.traj_idx % self.plot_interval == 0:
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.data[:, self.plot_idxs],
                    traj_idx=self.traj_idx,
                    update_buffer=True
                )
            # update cache
            self.io.update_cache(self.data)
            # update rewards
            self.Rs[self.traj_idx, 0] = self.R

        return self.Observations[self.T_step.shape[0] - 1], np.float_(self.R - prev_reward), terminated, truncated, {}
    
    def _update(self):
        """Method to update the trajectory data for the step.

        The first element at each time contains the current time.
        The next ``n_actions`` elements contain the actions.
        The next ``n_observations`` elements contain the observations.
        The next ``n_properties`` elements contain the properties.
        The final element is the cummulative reward from the step.
        """

        # update time
        _idxs = np.arange(self.t_idx, self.t_idx + self.T_step.shape[0])
        self.t_idx = _idxs[-1]
        self.t = self.T[_idxs[-1]]
        self.data[_idxs, 0] = self.T_step
        # update actions
        self.data[_idxs, 1:1 + self.n_actions] = self.actions
        # update observations
        self.data[_idxs, 1 + self.n_actions:1 + self.n_actions + self.n_observations] = self.Observations
        # update properties
        if self.save_properties:
            self.data[_idxs, 1 + self.n_actions + self.n_observations:-1] = np.array(self.get_Properties())
        # update reward
        self.data[_idxs, -1] = self.R + self.get_Reward()[-1]
        self.R = self.data[_idxs[-1], -1]
        # update plot
        if self.plot and not self.plot_interval:
            self.plotter.update_lines(
                y_js=self.data[_idxs, self.plot_idxs],
                j=_idxs[-1]
            )

    def evolve(self):
        """Method to freely evolve the trajectory."""

        # update actions
        self.actions = self.action_maxs

        # evolve
        for _ in tqdm(
            range(self.action_interval, self.t_dim, self.action_interval),
            desc="Progress (time)",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            # set evaluation times
            self.T_step = self.T[self.t_idx:self.t_idx + self.action_interval + 1] if self.t_idx + self.action_interval < self.t_dim else self.T[self.t_idx:]
            # step and update data
            self.Observations = self._step(
                actions=self.actions
            )
            self._update()

        # plot
        if self.plot and not self.plot_interval:
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=self.data[:, self.plot_idxs]
            )

    def close(self):
        """Method to close the environment."""

        # make replay gif
        self.plotter.make_gif(
            file_name=self.file_prefix + '_' + '_'.join([
                'replay',
                str(0),
                str(self.n_trajectories - 1),
                str(self.plot_interval)
            ])
        )
        # close plotter
        self.plotter.close()

        # close io
        self.io.close()

        # save learning curve
        self.plot_learning_curve(
            reward_data=self.Rs,
            axis_args=self.axis_args_learning_curve,
            save_plot=True,
            hold=False,
        )

        # clean
        del self.actions, self.Observations, self.Rs
        del self

    def replay_trajectories(self,
        idx_start:int=0,
        idx_end:int=None,
        plot_interval:int=0,
        make_gif:bool=True
    ):
        """Method to replay trajectories in a given range.
        
        Parameters
        ----------
        idx_start: int, default=0
            Starting index for the part file.
        idx_end: int, optional
            Ending index for the part file.
        plot_interval: int, default=0
            Number of trajectories after which the plots are updated. If non-positive, the environment's ``plot_interval`` value is taken.
        make_gif: bool, default=True
            Option to create a gif file for the replay.
        """

        # extract variables
        _idx_e = (idx_end) if idx_end is not None else (self.n_trajectories - 1)
        _interval = self.plot_interval if plot_interval <= 0 else plot_interval

        # get replay data in the given range
        replay_data = self.io.get_disk_cache(
            idx_start=idx_start,
            idx_end=_idx_e,
            idxs=self.plot_idxs
        )

        # update plotter
        for i in tqdm(
            range(0, replay_data.shape[0], _interval),
            desc="Plotting",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=replay_data[i],
                traj_idx=idx_start + i,
                update_buffer=True
            )
        # make gif
        if make_gif:
            self.plotter.make_gif(
                file_name=self.file_prefix + '_' + '_'.join([
                    'replay',
                    str(idx_start),
                    str(_idx_e),
                    str(_interval)
                ])
            )
        # hold plot
        self.plotter.hold_plot()

        # close plotter
        self.plotter.close()

    def plot_learning_curve(self,
        reward_data=None,
        axis_args:list=None,
        save_plot:bool=True,
        hold:bool=True
    ):
        """Method to plot the learning curve.

        Parameters
        ----------
        reward_data: :class:`numpy.ndarray`, default=None
            Cummulative rewards with shape ``(n_trajectories, 1)``. Loads data from disk cache if ``None``.
        axis_args: list, default=None
            Axis properties. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
        save_plot: bool, default=True
            Option to save the learning curve.
        hold: bool, default=True
            Option to hold the plot.
        """

        # close default plotter
        if getattr(self, 'plotter', None) is not None:
            self.plotter.close()

        # frequently used variables
        _idx_s = 0
        _idx_e = self.n_trajectories - 1

        # get reward trajectory data
        if reward_data is None:
            reward_data = self.io.get_disk_cache(
                idx_start=_idx_s,
                idx_end=_idx_e,
                idxs=[-1]
            )[:, -1, :]

        # new plotter
        plotter = TrajectoryPlotter(
            axes_args=[axis_args if axis_args is not None and len(axis_args) == 4 else self.axis_args_learning_curve],
            axes_lines_max=1,
            axes_cols=1,
            show_title=False
        )
        # plot lines
        plotter.plot_lines(
            xs=list(range(len(reward_data))),
            Y=reward_data
        )
        # save plot
        if save_plot:
            plotter.save_plot(
                file_name=self.file_prefix + '_' + '_'.join([
                    'learning_curve',
                    str(_idx_s),
                    str(_idx_e)
                ])
            )
        # hold plot
        if hold:
            plotter.hold_plot()

        # close plotter
        plotter.close()
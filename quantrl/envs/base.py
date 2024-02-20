#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base environments for reinforcement learning."""

__name__    = 'quantrl.envs.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-02-17"

# dependencies
from tqdm import tqdm
import gymnasium
import numpy as np

# quantrl modules
from ..io import FileIO
from ..plotters import TrajectoryPlotter

# TODO: Interface ConsoleIO

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
    n_observations: tuple
        Total number of observations.
    n_actions: tuple
        Total number of observations.
    action_maximums: list
        Maximum values of each action.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Available options are:
        ============================    ================================================
        key                             value
        ============================    ================================================
        has_delay                       (*bool*) option to implement delay functions. Default is ``False``.
        observation_space_range         (*list*) range of the observations. Default is ``[-1e9, 1e9]``.
        action_space_range              (*list*) range of the actions obtained from the network. The output is scaled by the corresponding action multiplier. Default is ``[-1.0, 1.0]``.
        action_space_type               (*str*) the type of action space. Options are ``"Binary"`` and ``"Box"``. Default is ``"Box"``.
        reward_max                      (*float*) maximum value of reward (implemented in children). Default is ``1.0``.
        reward_noise                    (*float*) noise in the reward function (implemented in children). Default is ``0.0``.
        plot                            (*bool*) option to plot the trajectories using ``:class:BaseTrajectoryPlotter``. Default is ``True``.
        plot_interval                   (*int*) number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
        plot_idxs                       (*list*) indices of the data values required to plot at each time step. Default is ``[-1]`` for the cummulative reward.
        axes_args                       (*list*) lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
        axes_lines_max                  (*int*) maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
        axes_cols                       (*int*) number of columns in the figure. Default is ``2``.
        max_trajectories_per_file       (*int*) maximum number of trajectory data to save per file. Default is ``100``.
        ============================    ================================================

    Notes
    -----
        The following required methods follow a strict formatting:
            ====================    ================================================
            method                  returns
            ====================    ================================================
            _step                   the updated observations with shape ``(action_interval + 1, n_observations)``, formatted as ``_step(actions)``, where ``actions`` is the array of actions with shape ``(n_actions, )`` multiplied by ``action_maximums``.
            get_Properties          the properties calculated from ``Observations`` with shape ``(action_interval + 1, n_properties)``.
            get_Reward              the reward calculated using the observations or the properties with shape ``(action_interval, )``. The class attributes ``reward_max`` and ``reward_noise`` can be utilized here.
            reset_Observations      ``None``. This method is used to reset the first entry (index ``0``) of the ``Observations`` variable which has shape ``(action_interval + 1, n_observations)`` with the initial values of the observations.
            ====================    ================================================
    """

    default_axis_args_learning_curve=['Episodes $N$', 'Cummulative Reward $\\tilde{R}$', [np.sqrt(10) * 1e-5, np.sqrt(10) * 1e4], 'log']
    """list: Default axis arguments to plot the learning curve."""

    default_kwargs = dict(
        has_delay=False,
        observation_space_range=[-1e9, 1e9],
        action_space_range=[-1.0, 1.0],
        action_space_type="Box",
        reward_max=1.0,
        reward_noise=0.0,
        plot=True,
        plot_interval=10,
        plot_idxs=[-1],
        axes_args=[
            ['$t / \\tau$', '$\\tilde{R}$', [np.sqrt(10) * 1e-5, np.sqrt(10) * 1e4], 'log']
        ],
        axes_lines_max=10,
        axes_cols=2,
        max_trajectories_per_file=100
    )
    """dict: Default values of all keyword arguments."""

    def __init__(self,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        dir_prefix='data',
        **kwargs
    ):
        """Class constructor for BaseGymEnv."""

        # update keyword arguments
        for key in self.default_kwargs:
            kwargs[key] = kwargs.get(key, self.default_kwargs[key])

        # validate
        assert t_norm_max > t_norm_ssz, 'maximum normalized time should be greater than the normalized step size'
        assert action_interval > 0, 'parameter ``action_interval`` should be a positive integer'
        assert kwargs['plot_interval'] >= 0, 'parameter ``plot_interval`` should be a non-negative integer'
        assert kwargs['plot_interval'] < n_trajectories if kwargs['plot'] else True, 'parameter ``plot_interval`` should be a less than parameter ``n_trajectories``'
        assert len(kwargs['plot_idxs']) == len(kwargs['axes_args']), 'number of indices for plot should match number of axes arguments'

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
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.action_maximums = np.array(action_maximums, dtype=np.float_)
        self.action_interval = action_interval
        # align delay with action interval
        self.has_delay = kwargs['has_delay']
        self.t_delay = self.T[self.action_interval] - self.T[0]
        # extend one step if not divisible
        _action_dim = (self.t_dim - 1) / self.action_interval
        self.total_timesteps = self.n_trajectories * (int(_action_dim + 1) if _action_dim - int(_action_dim) > 0 else int(_action_dim))
        # reward constants
        self.reward_max = np.float_(kwargs['reward_max'])
        self.reward_noise = np.float_(kwargs['reward_noise'])
        # data constants
        self.has_properties = False
        self.file_prefix = dir_prefix + '_' + '_'.join([
            str(self.n_trajectories),
            str(self.t_norm_max),
            str(self.t_norm_ssz),
            str(self.t_norm_mul),
            str(self.action_maximums),
            str(self.action_interval)]
        ) + '/env'
        # plot constants
        self.plot = kwargs['plot']
        self.plot_interval = kwargs['plot_interval']
        self.plot_idxs = kwargs['plot_idxs']

        # initialize Gymnasium environment
        super().__init__()
        # discrete actions
        if "Binary" in kwargs['action_space_type']:
            self.action_space_range = [0, 1]
            self.action_space = gymnasium.spaces.MultiDiscrete(
                nvec=[2] * self.n_actions,
            )
        # continuous actions
        else:
            self.action_space_range = kwargs['action_space_range']
            self.action_space = gymnasium.spaces.Box(
                low=self.action_space_range[0],
                high=self.action_space_range[1],
                shape=(self.n_actions, ),
                dtype=np.float_
            )
        # continuous observations
        self.observation_space_range = kwargs['observation_space_range']
        self.observation_space = gymnasium.spaces.Box(
            low=self.observation_space_range[0],
            high=self.observation_space_range[1],
            shape=(self.n_observations, ),
            dtype=np.float_
        )

        # initialize IO
        self.io = FileIO(
            disk_cache_dir=self.file_prefix + '_cache',
            max_cache_size=kwargs['max_trajectories_per_file']
        )

        # initialize plotter
        if self.plot:
            self.plotter = TrajectoryPlotter(
                axes_args=kwargs['axes_args'],
                axes_lines_max=kwargs['axes_lines_max'],
                axes_cols=kwargs['axes_cols'],
                show_title=True
            )

        # initialize buffers
        self.traj_idx = -1
        self.actions = np.zeros(self.n_actions, dtype=np.float_)
        self.Observations = np.zeros((self.action_interval + 1, self.n_observations), dtype=np.float_)
        self.Rs = np.zeros((self.n_trajectories, 1), dtype=np.float_)

        # validate child environment
        try:
            self.reset_Observations()
            self.Observations[1:] = self.Observations[0]
            self.n_data_elements = self.n_actions + self.n_observations + 2
            if getattr(self, 'get_Properties', None) is not None:
                n_properties = np.shape(self.get_Properties())[1]
                self.n_data_elements += n_properties
                self.has_properties = True
            self.R = self.get_Reward()[-1]
        except AttributeError as error:
            print(f"Missing required method or attribute: ({error}). Refer to **Notes** of :class:`quantrl.envs.base.BaseGymEnv` for the implementation format of the missing method or add the missing attribute to the ``reset_Observations`` method.")
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
            Inititial observations.
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
            Actions at current time.

        Returns
        -------
        observations: :class:`numpy.ndarray`
            Observations at current time.
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
        self.actions = actions * self.action_maximums
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
        if self.has_properties:
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
        self.actions = self.action_maximums

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
        if self.plot and self.plot_interval:
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
            axis_args=self.default_axis_args_learning_curve,
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
            axes_args=[axis_args if axis_args is not None and len(axis_args) == 4 else self.default_axis_args_learning_curve],
            axes_lines_max=1,
            axes_cols=1,
            show_title=False
        )
        # plot lines
        plotter.plot_lines(
            xs=list(range(reward_data.shape[0])),
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
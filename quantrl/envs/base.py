#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base environments for reinforcement learning."""

__name__    = 'quantrl.envs.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-03-01"

# dependencies
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3.common import env_util
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm
import numpy as np

# quantrl modules
from ..io import FileIO
from ..plotters import TrajectoryPlotter

# TODO: Interface ConsoleIO
# TODO: Optimize BaseVecEnv truncation
# TODO: Separate state and observation data

class BaseEnv():
    r"""Base environment for reinforcement-learning.

    Initializes ``T_norm``, ``T``, ``observation_space``, ``action_space``, ``action_steps``, ``file_path_prefix``, ``io`` and ``plotter``.

    The interfaced environment needs to implement ``_step``, ``reset_observations``, ``get_properties`` and ``get_reward`` methods.
    Refer to **Notes** below for their implementations.

    Parameters
    ----------
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_observations: int
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: int
        Total number of actions.
    action_maximums: list
        Maximum values of each action with shape ``(n_actions, )``.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    file_prefix: str, default='base_env'
        Prefix of the files where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Available options are:
        ========================    ================================================
        key                         value
        ========================    ================================================
        has_delay                   (*bool*) option to implement delay functions. Default is ``False``.
        observation_space_range     (*list*) range of the observations. Default is ``[-1e9, 1e9]``.
        action_space_range          (*list*) range of the actions. The output is scaled by the corresponding action multiplier. Default is ``[-1.0, 1.0]``.
        action_space_type           (*str*) the type of action space. Options are ``'binary'`` and ``'box'``. Default is ``'box``.
        reward_max                  (*float*) maximum value of reward (implemented in children). Default is ``1.0``.
        reward_noise                (*float*) noise in the reward function (implemented in children). Default is ``0.0``.
        disk_cache_size             (*int*) number of environments to save per disk-cache. Default is ``100``.
        average_over                (*int*) number of episodes to run the running average over. This value should be less than or equal to the total number of episodes. Default is ``100``.
        plot                        (*bool*) option to plot the trajectories using ``:class:BaseTrajectoryPlotter``. Default is ``True``.
        plot_interval               (*int*) number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
        plot_idxs                   (*list*) indices of the data values required to plot at each time step. Default is ``[-1]`` for the cummulative reward.
        axes_args                   (*list*) lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
        axes_lines_max              (*int*) maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
        axes_cols                   (*int*) number of columns in the figure. Default is ``2``.
        ========================    ================================================

    Notes
    -----
        The following required methods follow a strict formatting:
            ====================    ================================================
            method                  returns
            ====================    ================================================
            reset_observations      the initial observations with shape either ``(n_observations, )`` or ``(n_envs, n_observations)``. These observations are then internally initialized to the last element of ``Observations`` with shape ``(action_interval + 1, n_observations, )`` or ``(action_interval + 1, n_envs, n_observations)``
            get_properties          the properties calculated from ``Observations`` with shape either ``(action_interval + 1, n_properties)`` or ``(action_interval + 1, n_envs, n_properties)``.
            get_reward              the reward calculated using the observations or the properties with shape either ``(action_interval + 1, )`` or ``(action_interval + 1, n_envs)``. The class attributes ``reward_max`` and ``reward_noise`` can be utilized here.
            _step                   the updated observations with shape either ``(action_interval + 1, n_observations)`` or ``(action_interval + 1, n_envs, n_observations), formatted as ``_step(actions)``, where ``actions`` is the array of actions with shape either ``(n_actions, )`` or ``(n_envs, n_actions)`` multiplied by ``action_maximums``. This method is already implemented by the child classes in :class:`quantrl.envs.deterministic` and :class:`quantrl.envs.stochastic`.
            ====================    ================================================
    """

    default_axis_args_learning_curve=['Episodes', 'Average Return', [np.sqrt(10) * 1e-4, np.sqrt(10) * 1e6], 'log']
    """list: Default axis arguments to plot the learning curve."""

    base_env_kwargs = dict(
        has_delay=False,
        observation_space_range=[-1e9, 1e9],
        action_space_range=[-1.0, 1.0],
        action_space_type='box',
        reward_max=1.0,
        reward_noise=0.0,
        disk_cache_size=100,
        average_over=100,
        plot=True,
        plot_interval=10,
        plot_idxs=[-1],
        axes_args=[
            ['$t / \\tau$', '$\\tilde{R}$', [np.sqrt(10) * 1e-5, np.sqrt(10) * 1e4], 'log']
        ],
        axes_lines_max=10,
        axes_cols=2
    )
    """dict: Default values of all keyword arguments."""

    def __init__(self,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        dir_prefix:str='data',
        file_prefix:str='base_env',
        **kwargs
    ):
        """Class constructor for BaseEnv."""

        # update keyword arguments
        for key in self.base_env_kwargs:
            kwargs[key] = kwargs.get(key, self.base_env_kwargs[key])

        # validate arguments
        assert t_norm_max > t_norm_ssz, "maximum normalized time should be greater than the normalized step size"
        assert action_interval > 0, "parameter ``action_interval`` should be a positive integer"
        assert kwargs['plot_interval'] >= 0, "parameter ``plot_interval`` should be a non-negative integer"
        assert len(kwargs['plot_idxs']) == len(kwargs['axes_args']), "number of indices for plot should match number of axes arguments"
        assert len(kwargs['observation_space_range']) == 2, "parameter ``observation_space_range`` should contain two elements for the minimum and maximum values, both inclusive"
        assert len(kwargs['action_space_range']) == 2, "parameter ``action_space_range`` should contain two elements for the minimum and maximum values, both inclusive"
        assert kwargs['action_space_type'] in ['binary', 'box'], "parameter ``action_space_type`` can be either ``'binary'`` or ``'box'``"
        assert kwargs['disk_cache_size'] > 0, "parameter ``disk_cache_size`` should be a positive integer"

        # time attributes
        self.t_norm_max = t_norm_max
        self.t_norm_ssz = t_norm_ssz
        self.t_norm_mul = t_norm_mul
        # truncate before maximum time if not divisible
        _shape_T = (int(self.t_norm_max / self.t_norm_ssz) + 1, )
        self.T_norm = np.arange(_shape_T[0], dtype=np.float_) * self.t_norm_ssz
        self.T = self.T_norm * t_norm_mul

        # observation and property attributes
        self.n_observations = n_observations
        self.observation_space_range = kwargs['observation_space_range']
        self.observation_space = Box(
            low=self.observation_space_range[0],
            high=self.observation_space_range[1],
            shape=(self.n_observations, ),
            dtype=np.float_
        )
        self.n_properties = n_properties

        # action attributes
        self.n_actions = n_actions
        self.action_space_type = kwargs['action_space_type']
        # discrete actions
        if self.action_space_type == 'binary':
            self.action_space_range = [0, 1]
            self.action_space = MultiDiscrete(
                nvec=[2] * self.n_actions,
            )
        # continuous actions
        else:
            self.action_space_range = kwargs['action_space_range']
            self.action_space = Box(
                low=self.action_space_range[0],
                high=self.action_space_range[1],
                shape=(self.n_actions, ),
                dtype=np.float_
            )
        self.action_maximums = np.array(action_maximums, dtype=np.float_)
        self.action_interval = action_interval
        self.action_steps = int(np.ceil((self.T.shape[0] - 1) / self.action_interval))

        # align delay with action interval
        self.has_delay = kwargs['has_delay']
        self.t_delay = self.T[self.action_interval] - self.T[0]
        # reward constants
        self.reward_max = np.float_(kwargs['reward_max'])
        self.reward_noise = np.float_(kwargs['reward_noise'])

        # data constants
        self.dir_path = dir_prefix + '/' + '_'.join([
            str(self.t_norm_max),
            str(self.t_norm_ssz),
            str(self.t_norm_mul),
            str(self.action_maximums),
            str(self.action_interval)
        ])
        self.file_path_prefix = self.dir_path + '/' + file_prefix
        self.n_data = 1 + self.n_actions + self.n_observations + self.n_properties + 1
        self.average_over = int(kwargs['average_over'])

        # initialize IO
        self.io = FileIO(
            data_shape=(self.T.shape[0], self.n_data),
            disk_cache_dir=self.file_path_prefix + '_cache',
            disk_cache_size=kwargs['disk_cache_size']
        )
        
        # plot constants
        self.plot = kwargs['plot']
        self.plot_interval = kwargs['plot_interval']
        self.plot_idxs = kwargs['plot_idxs']
        # initialize plotter
        if self.plot:
            self.plotter = TrajectoryPlotter(
                axes_args=kwargs['axes_args'],
                axes_lines_max=kwargs['axes_lines_max'],
                axes_cols=kwargs['axes_cols'],
                show_title=True,
                save_dir=self.file_path_prefix + '_plots'
            )

        # initialize buffers
        self.t_idx = 0
        self.t = np.float_(0.0)

    def validate_environment(self,
        shape_reset_observations:tuple,
        shape_get_properties:tuple,
        shape_get_reward:tuple
    ):
        """Method to validate the interfaced environment.
        
        Parameters
        ----------
        shape_reset_observations: tuple
            Shape of the array returned by ``reset_observations``.
        shape_get_properties: tuple
            Shape of the array returned by ``get_properties``.
        shape_get_reward: tuple
            Shape of the array returned by ``get_reward``.
        """

        try:
            # validate initial observations
            observations_0 =self.reset_observations()
            assert np.shape(observations_0) == shape_reset_observations, "``reset_observations`` should return a 2D array with shape ``{}``".format(shape_reset_observations)
            # initialize observations for each time step
            self.Observations = np.array([observations_0]).repeat(self.action_interval + 1, axis=0)
            # validate properties
            if self.n_properties > 0:
                self.Properties = np.array(self.get_properties())
                assert self.Properties.shape == shape_get_properties, "``get_properties`` should return a 2D array with shape ``{}``".format(shape_get_properties)
            # validate reward
            self.Reward = np.array(self.get_reward())
            assert self.Reward.shape == shape_get_reward, "``get_reward`` should return a 1D array with shape ``{}``".format(shape_get_reward)
        except AttributeError as error:
            print(f"Missing required method or attribute: ({error}). Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for the implementation format of the missing method or add the missing attribute to the ``reset_observations`` method.")
            exit()

    def reset(self):
        """Method to reset the time and obtain initial observations.
        
        Returns
        -------
        observations_0: :class:`numpy.ndarray`
            Inititial observations.
        """

        # reset time
        self.t_idx = 0
        self.t = np.float_(0.0)
        
        # update observations
        observations_0 = np.array(self.reset_observations())
        self.Observations[-1] = observations_0

        return observations_0

    def update(self):
        """Method to update the time, observations, properties and reward and obtain the final set of observations and reward.

        Returns
        -------
        observations: :class:`numpy.ndarray`
            Observations at current time.
        reward: float
            Reward calculated.
        terminated: bool
            Flag to terminate trajectory.
        """

        # set evaluation times
        self.T_step = self.T[self.t_idx:self.t_idx + self.action_interval + 1] if self.t_idx + self.action_interval < self.T.shape[0] else self.T[self.t_idx:]

        # step and update observations
        self.Observations = self._step()

        # update properties
        if self.n_properties > 0:
            self.Properties = np.array(self.get_properties())

        # update rewards
        self.Reward = np.array(self.get_reward())

        # update time
        self.t_idx += self.T_step.shape[0] - 1
        self.t = self.T[self.t_idx]

        # check if completed
        terminated = False if self.t_idx + 1 < self.T.shape[0] else True

        return self.Observations[self.T_step.shape[0] - 1], self.Reward[self.T_step.shape[0] - 1], terminated

    def plot_learning_curve(self,
        reward_data=None,
        n_episodes:int=None,
        axis_args:list=None,
        save_plot:bool=True,
        hold:bool=True
    ):
        """Method to plot the learning curve.

        Either one of the parameters ``n_episodes`` or ``reward_data`` should be provided.

        Parameters
        ----------
        reward_data: :class:`numpy.ndarray`, default=None
            Cummulative rewards with shape ``(n_trajectories, 1)``. Loads data from disk cache if ``None``.
        n_episodes: int, default=None
            Total number of episodes to load from cache.
        axis_args: list, default=None
            Axis properties. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
        save_plot: bool, default=True
            Option to save the learning curve.
        hold: bool, default=True
            Option to hold the plot.
        """

        # validate arguments
        assert reward_data is not None or n_episodes is not None, "either one of the parameters ``n_episodes`` or ``reward_data`` should be provided"

        # extract frequently used variables
        _idx_s = 0

        # get reward trajectory data
        if reward_data is None:
            _idx_e = n_episodes - 1
            reward_data = self.io.get_disk_cache(
                idx_start=_idx_s,
                idx_end=_idx_e,
                idxs=[-1]
            )[:, -1, :]
        else:
            _idx_e = reward_data.shape[0] - 1

        # calculate average return and its standard deviation
        assert reward_data.shape[0] >= self.average_over, "parameter ``average_over`` should be less than or equal to the total number of episodes"
        _shape = reward_data.shape
        return_avg = np.convolve(reward_data[:, 0], np.ones((self.average_over, )) / np.float_(self.average_over), mode='same').reshape(_shape)

        # new plotter
        plotter = TrajectoryPlotter(
            axes_args=[axis_args if axis_args is not None and len(axis_args) == 4 else self.default_axis_args_learning_curve],
            axes_lines_max=3,
            axes_cols=1,
            show_title=False
        )
        # plot reward data
        plotter.plot_lines(
            xs=list(range(_shape[0])),
            Y=reward_data
        )
        # plot average return
        plotter.plot_lines(
            xs=list(range(_shape[0])),
            Y=return_avg
        )
        # save plot
        if save_plot:
            plotter.save_plot(
                file_name=self.file_path_prefix + '_' + '_'.join([
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

    def replay_trajectories(self,
        n_episodes,
        idx_start:int=0,
        plot_interval:int=0,
        make_gif:bool=True
    ):
        """Method to replay trajectories in a given range.
        
        Parameters
        ----------
        n_episodes: int
            Total number of episodes.
        idx_start: int, default=0
            Starting index for the cached files.
        plot_interval: int, default=0
            Number of trajectories after which the plots are updated. If non-positive, the environment's ``plot_interval`` value is taken.
        make_gif: bool, default=True
            Option to create a gif file for the replay.
        """

        # extract frequently used variables
        _idx_e = n_episodes - 1
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
                file_name=self.file_path_prefix + '_' + '_'.join([
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

    def close(self,
        n_episodes,
        save_replay=True
    ):
        """Method to close the environment.
        
        Parameters
        ----------
        n_episodes: int
            Total number of episodes.
        save_replay: bool, default=True
            Option to save the replay as gif.
        """

        if self.plot and save_replay:
            # make replay gif
            self.plotter.make_gif(
                file_name=self.file_path_prefix + '_' + '_'.join([
                    'replay',
                    str(0),
                    str(n_episodes - 1),
                    str(self.plot_interval)
                ])
            )
            # close plotter
            self.plotter.close()

        # clean
        del self.T, self.T_norm, self.T_step, self.actions, self.Observations, self.Reward
        if self.n_properties > 0:
            del self.Properties
        del self

class BaseGymEnv(BaseEnv, Env):
    r"""Gymnasium-based environment for reinforcement-learning.

    Initializes ``action_space``, ``observation_space`` and ``io``, whereas ``T_norm``, ``T``, ``file_path_prefix`` and ``plotter`` are initialized by :class:`quantrl.envs.base.BaseEnv`.

    The interfaced environment needs to implement ``_step``, ``reset_observations``, ``get_properties`` and ``get_reward`` methods.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.

    Parameters
    ----------
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_observations: int
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: int
        Total number of actions.
    action_maximums: list
        Maximum values of each action.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    file_prefix: str, default='base_gym_env'
        Prefix of the files where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseEnv` for available options.
    """

    def __init__(self,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        dir_prefix='data',
        file_prefix='base_gym_env',
        **kwargs
    ):
        """Class constructor for BaseGymEnv."""

        # initialize BaseEnv
        super().__init__(
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            dir_prefix=dir_prefix,
            file_prefix=file_prefix,
            **kwargs
        )

        # validate interfaced environment
        self.validate_environment(
            shape_reset_observations=(self.n_observations, ),
            shape_get_properties=(self.action_interval + 1, self.n_properties),
            shape_get_reward=(self.action_interval + 1, )
        )

        # initialize Gymnasium environment
        Env.__init__(self)

        # initialize buffers
        self.Rs = list()
        self.traj_idx = -1
        self.R = 0.0
        self.data = np.zeros((self.T.shape[0], self.n_data), dtype=np.float_)

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

        # initialize reset
        observations_0 = super().reset()

        # update buffers
        self.traj_idx += 1
        self.R = 0.0
        self.data = np.zeros((self.T.shape[0], self.n_data), dtype=np.float_)

        return observations_0, {
            'traj_idx': self.traj_idx
        }

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
            Reward calculated.
        terminated: bool
            Flag to terminate trajectory.
        truncated: bool
            Flag to truncate trajectory.
        info: dict
            Additional information.
        """

        # set actions
        self.actions = actions * self.action_maximums

        # get observations, properties and reward
        observations, reward, terminated = super().update()

        # update reward
        self.R += reward
        # update data
        truncated = self.update_data()

        # if trajectory ends
        if terminated or truncated:
            # update plotter and io
            if self.plot and self.plot_interval and self.traj_idx % self.plot_interval == 0:
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.data[:, self.plot_idxs],
                    traj_idx=self.traj_idx,
                    update_buffer=True
                )
            self.io.update_cache(
                data=self.data
            )
            # update episode reward
            self.Rs.append(self.R)

        return observations, reward, terminated, truncated, {}

    def update_data(self):
        """Method to update the trajectory data for the step.

        The first element at each time contains the current time.
        The next ``n_actions`` elements contain the actions.
        The next ``n_observations`` elements contain the observations.
        The next ``n_properties`` elements contain the properties.
        The final element is the cummulative reward from the step.
        """

        # update data
        _idxs = np.arange(self.t_idx - self.T_step.shape[0] + 1, self.t_idx + 1)
        self.data[_idxs, 0] = self.T_step
        self.data[_idxs, 1:1 + self.n_actions] = self.actions
        self.data[_idxs, 1 + self.n_actions:1 + self.n_actions + self.n_observations] = self.Observations
        if self.n_properties > 0:
            self.data[_idxs, 1 + self.n_actions + self.n_observations:-1] = self.Properties
        self.data[_idxs, -1] = self.R
        
        # check if out of bounds
        truncated = np.max(self.Observations) > self.observation_space_range[1] or np.min(self.Observations) < self.observation_space_range[0]
        print(f'Trajectory #{self.traj_idx} truncated') if truncated else True

        return truncated

    def evolve(self,
        close=True
    ):
        """Method to freely evolve the trajectory.
        
        Parameters
        ----------
        close: bool, default=True
            Option to close the environment.
        """

        # evolve
        for _ in tqdm(
            range(self.action_steps),
            desc="Progress (time)",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            # set actions
            self.actions = self.action_maximums

            # step
            _, reward, _ = super().update()

            # update reward
            self.R += reward
            # update data
            truncated = self.update_data()

            # stop if out of bounds
            if truncated:
                break

        # plot
        if self.plot and self.plot_interval:
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=self.data[:, self.plot_idxs]
            )
            self.plotter.hold_plot()

        # close environment
        if close:
            self.close(
                save=False
            )

    def close(self,
        save=True
    ):
        """Method to close the environment.
        
        Parameters
        ----------
        save: bool, default=True
            Option to save the learning curve and replay.
        """

        if save:
            # save learning curve
            self.plot_learning_curve(
                reward_data=np.array(self.Rs).reshape((len(self.Rs), 1)),
                hold=False
            )
        del self.Rs

        # close io
        self.io.close(
            dump_cache=save
        )

        # clean
        super().close(
            n_episodes=self.traj_idx,
            save_replay=save
        )

class BaseSB3Env(BaseEnv, VecEnv):
    r"""Stable-Baselines3-based vectorized environments for reinforcement-learning.

    Initializes ``action_space``, ``observation_space`` and ``io``, whereas ``T_norm``, ``T``, ``file_path_prefix`` and ``plotter`` are initialized by :class:`quantrl.envs.base.BaseEnv`.

    The interfaced environment needs to implement ``_step``, ``reset_observations``, ``get_properties`` and ``get_reward`` methods.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.

    Parameters
    ----------
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_envs: int
        Total number of environments to run in parallel.
    n_observations: int
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: int
        Total number of actions.
    action_maximums: list
        Maximum values of each action.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    file_prefix: str, default='base_vec_env'
        Prefix of the files where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseEnv` for available options. Here, the parameter ``disk_cache_size`` is set to ``n_envs``.
    """

    def __init__(self,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_envs:int,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        dir_prefix='data',
        file_prefix='base_vec_env',
        **kwargs
    ):
        """Class constructor for BaseSB3Env."""

        # initialize BaseEnv
        super().__init__(
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            dir_prefix=dir_prefix,
            file_prefix=file_prefix,
            disk_cache_size=n_envs,
            **kwargs
        )

        # update attributes
        self.n_envs = n_envs
        self.render_mode = [None] * n_envs
        self.action_maximums_batch = np.array([self.action_maximums]).repeat(self.n_envs, axis=0)

        # validate interfaced environment
        self.validate_environment(
            shape_reset_observations=(self.n_envs, self.n_observations),
            shape_get_properties=(self.action_interval + 1, self.n_envs, self.n_properties),
            shape_get_reward=(self.action_interval + 1, self.n_envs)
        )

        # initialize SB3 environment
        VecEnv.__init__(self,
            num_envs=self.n_envs,
            observation_space=self.observation_space,
            action_space=self.action_space
        )

        # initialize buffers
        self.Rs = list()
        self.batch_idx = -1
        self.R = np.zeros(self.n_envs, dtype=np.float_)
        self.data = np.zeros((self.T.shape[0], self.n_envs, self.n_data), dtype=np.float_)

    def env_is_wrapped(self,
        wrapper_class,
        indices=None
    ):
        """Method to check if a batch of sub-environments are wrapped with the given wrapper.
        
        Parameters
        ----------
        wrapper_class: :class:`gymnasium.Wrapper`
            Wrapper class.
        indices: int or list, default=None
            Indices of the environments. If ``None``, the values for all sub-environments are considered.

        Returns
        -------
        wrapped: list
            Whether the batch of sub-environments are wrapped.
        """

        return [env_util.is_wrapped(self, wrapper_class) for _ in range(indices if indices is not None else self.n_envs)]
    
    def env_method(self,
        method_name,
        *method_args,
        indices=None,
        **method_kwargs
    ):
        """Method to call other methods of the sub-environments.

        Parameters
        ----------
        method_name: str
            Name of the method.
        method_args: tuple
            Additional positional arguments.
        indices: int or list, default=None
            Indices of the environments. If ``None``, the values for all sub-environments are considered.
        method_kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        method: list
            Methods of the sub-environments.
        """

        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(indices if indices is not None else self.n_envs)]
    
    def get_attr(self,
        attr_name,
        indices=None         
    ):
        """Method to obtain attributes of the sub-environments.

        Parameters
        ----------
        attr_name: str
            Name of the attribute.
        indices: int or list, default=None
            Indices of the environments. If ``None``, the values for all sub-environments are considered.

        Returns
        -------
        attribute: list
            Attributes of the sub-environments.
        """

        return [getattr(self, attr_name) for _ in range(indices if indices is not None else self.n_envs)]
    
    def set_attr(self,
        attr_name,
        value,
        indices=None
    ):
        """Method to assign attributes of the sub-environments.

        Parameters
        ----------
        attr_name: str
            Name of the attribute.
        value: any
            Value of the attribute.
        indices: int or list, default=None
            Indices of the environments. If ``None``, the values for all sub-environments are considered.
        """

        [setattr(self, attr_name, value) for _ in range(indices if indices is not None else self.n_envs)]

    def reset(self,
        seed:float=None,
        options:dict=None
    ):
        """Method to reset all variables for a new batch.

        Parameters
        ----------
        seed: float
            Seed value for the reset.
        options: dict
            Options for the reset.
        
        Returns
        -------
        observations_0: :class:`numpy.ndarray`
            Inititial observations.
        info: str
            Information on the reset.
        """

        # initialize reset
        observations_0 = super().reset()

        # update buffers
        self.batch_idx += 1
        self.R = np.zeros(self.n_envs, dtype=np.float_)
        self.data = np.zeros((self.T.shape[0], self.n_envs, self.n_data), dtype=np.float_)

        return observations_0
    
    def step_async(self,
        actions
    ):
        """Method to prepare for one single step.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Actions at current time.
        """

        # set actions
        self.actions = actions * self.action_maximums_batch

    def step_wait(self):
        """Method to take one single step and wait for the result.

        Returns
        -------
        observations: :class:`numpy.ndarray`
            Observations at current time.
        reward: float
            Reward calculated for the action interval.
        terminated: bool
            Flag to terminate trajectory.
        info: dict
            Additional information.
        """

        # get observations, properties and reward
        observations, reward, terminated = super().update()

        # update reward
        self.R += reward
        # update data
        truncated = self.update_data()

        infos = [{}] * self.n_envs
        # if trajectory ends
        if terminated or truncated:
            # update plotter and io
            if self.plot and self.plot_interval:
                _arr = np.arange(self.n_envs, dtype=np.int_)
                _idxs = _arr[(self.batch_idx * self.n_envs + _arr) % self.plot_interval == 0]
                for _idx in _idxs:
                    self.plotter.plot_lines(
                        xs=self.T_norm,
                        Y=self.data[:, _idx, self.plot_idxs],
                        traj_idx=self.batch_idx * self.n_envs + _idx,
                        update_buffer=True
                    )
            self.io.disk_cache(
                data=self.data,
                batch_idx=self.batch_idx
            )
            # update episode reward
            self.Rs.append(self.R)

            # update info
            infos = [{
                "Timelimit.truncated": True,
                "terminal_observation": obs
            } for obs in observations]

            # reset environment
            observations = self.reset()

        return observations, reward, [terminated or truncated] * self.n_envs, infos

    def update_data(self):
        """Method to update the batch data for the step.

        The first element at each time contains the batch of stepped time.
        The next ``n_actions`` elements contain the batch of actions.
        The next ``n_observations`` elements contain the batch of observations.
        The next ``n_properties`` elements contain the batch of properties.
        The final element is the batch of cummulative reward from the step.
        """

        # update data
        _idxs = np.arange(self.t_idx - self.T_step.shape[0] + 1, self.t_idx + 1)
        self.data[_idxs, :, 0] = self.T_step.repeat(self.n_envs).reshape((self.T_step.shape[0], self.n_envs))
        self.data[_idxs, :,  1:1 + self.n_actions] = self.actions
        self.data[_idxs, :, 1 + self.n_actions:1 + self.n_actions + self.n_observations] = self.Observations
        if self.n_properties > 0:
            self.data[_idxs, :, 1 + self.n_actions + self.n_observations:-1] = self.Properties
        self.data[_idxs, :, -1] = self.R

        # check if out of bounds
        truncated = np.max(self.Observations) > self.observation_space_range[1] or np.min(self.Observations) < self.observation_space_range[0]
        print(f'Batch #{self.batch_idx} truncated') if truncated > 0 else True

        return truncated

    def evolve(self,
        close=True,
        save=False
    ):
        """Method to freely evolve the trajectory.
        
        Parameters
        ----------
        close: bool, default=True
            Option to close the environment.
        save: bool, default=False
            Option to save the learning curve and replay.
        """

        # evolve
        for _ in tqdm(
            range(self.action_steps),
            desc="Progress (time)",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            # set actions
            self.actions = self.action_maximums_batch

            # udpate reward
            _, reward, _ = super().update()

            # update reward
            self.R += reward
            # update data
            truncated = self.update_data()

            # stop if out of bounds
            if truncated:
                break
            
        # update episode reward
        self.Rs.append(self.R)

        # plot
        if self.plot and self.plot_interval:
            _arr = np.arange(self.n_envs, dtype=np.int_)
            _idxs = _arr[_arr % self.plot_interval == 0]
            for _idx in _idxs:
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.data[:, _idx, self.plot_idxs],
                    traj_idx=_idx,
                    update_buffer=True
                )
            self.plotter.hold_plot()

        # close environment
        if close:
            self.close(
                save=save
            )

    def close(self,
        save=True
    ):
        """Method to close the environment.
        
        Parameters
        ----------
        save: bool, default=True
            Option to save the learning curve and replay.
        """

        if save:
            # save learning curve
            self.plot_learning_curve(
                reward_data=np.array(self.Rs).reshape((len(self.Rs) * self.n_envs, 1)),
                hold=False
            )
        del self.Rs

        # close io
        self.io.close(
            dump_cache=save
        )

        # clean
        super().close(
            n_episodes=self.batch_idx,
            save_replay=save
        )
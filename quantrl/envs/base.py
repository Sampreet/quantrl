#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base environments for reinforcement learning."""

__name__    = 'quantrl.envs.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-07-22"

# dependencies
from abc import ABC, abstractmethod
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3.common import env_util
from stable_baselines3.common.vec_env import VecEnv
from tqdm.rich import tqdm
import numpy as np

# quantrl modules
from ..backends.base import BaseBackend
from ..io import FileIO
from ..plotters import TrajectoryPlotter, LearningCurvePlotter

# TODO: Interface ConsoleIO
# TODO: Support for different number of states and observables

class BaseEnv(ABC):
    r"""Base environment for reinforcement-learning.

    Initializes ``T_norm``, ``T``, ``observation_space``, ``action_space``, ``action_steps``, ``file_path_prefix``, ``io`` and ``plotter``.

    The interfaced environment needs to implement ``_update_states``, ``reset_states``, and ``get_reward`` methods.
    Additionally, the ``get_properties`` method should be overridden if ``n_properties`` is non-zero.

    Parameters
    ----------
    backend: :class:`quantrl.backends.*`
        Backend to use for numerical computations.
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
    data_idxs: list
        Indices of the data to store into the ``data`` attribute. The indices can be selected from the complete set of values at each point of time (total ``1 + n_actions + n_observations + n_properties + 1`` elements in the same order, where the first element is the time and the last element is the reward).
    dir_prefix: str
        Prefix of the directory where the data will be stored.
    file_prefix: str
        Prefix of the files where the data will be stored.
    kwargs: dict
        Keyword arguments. Available options are:
        
        ========================    ================================================
        key                         value
        ========================    ================================================
        has_delay                   (*bool*) option to implement delay functions. Default is ``False``.
        observation_space_range     (*list*) range of the observations. Default is ``[-1e12, 1e12]``.
        observation_stds            (*list* or ``None``) standard deviations of the observed states from the actual states. Default is ``None``.
        action_space_range          (*list*) range of the actions. The output is scaled by the corresponding action multiplier. Default is ``[-1.0, 1.0]``.
        action_space_type           (*str*) the type of action space. Options are ``'binary'`` and ``'box'``. Default is ``'box``.
        seed                        (*int*) seed to initialize random number generators. If ``None``, a random integer seed is generated. Default is ``None``.
        cache_all_data              (*bool*) option to cache all data to disk. If ``False``, only the indices of ``data_idxs`` are stored. Default is ``True``.
        cache_dump_interval         (*int*) number of environments to cache before dumping to disk. Default is ``100``.
        average_over                (*int*) number of episodes to run the running average over. This value should be less than or equal to the total number of episodes. Default is ``100``.
        plot                        (*bool*) option to plot the trajectories using ``:class:BaseTrajectoryPlotter``. Default is ``True``.
        plot_interval               (*int*) number of trajectories after which the plots are updated. Must be a positive integer. Default is ``10``.
        plot_idxs                   (*list*) indices of the data values required to plot at each time step. Default is ``[-1]`` for the cummulative reward.
        axes_args                   (*list*) lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
        axes_lines_max              (*int*) maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``10``.
        axes_cols                   (*int*) number of columns in the figure. Default is ``2``.
        plot_buffer                 (*bool*) option to store a buffer of plots for to make a gif file. Default is ``False``.
        ========================    ================================================
    """

    default_axis_args_learning_curve=['Episodes', 'Average Return', [np.sqrt(10) * 1e-4, np.sqrt(10) * 1e6], 'log']
    """list: Default axis arguments to plot the learning curve."""

    base_env_kwargs = dict(
        has_delay=False,
        observation_space_range=[-1e12, 1e12],
        observation_stds=None,
        action_space_range=[-1.0, 1.0],
        action_space_type='box',
        seed=None,
        cache_all_data=True,
        cache_dump_interval=100,
        average_over=100,
        plot=True,
        plot_interval=10,
        plot_idxs=[-1],
        axes_args=[
            ['$t / \\tau$', '$\\tilde{R}$', [np.sqrt(10) * 1e-5, np.sqrt(10) * 1e4], 'log']
        ],
        axes_lines_max=10,
        axes_cols=2,
        plot_buffer=False
    )
    """dict: Default values of all keyword arguments."""

    def __init__(self,
        backend:BaseBackend,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        data_idxs:list,
        dir_prefix:str,
        file_prefix:str,
        **kwargs
    ):
        """Class constructor for BaseEnv."""

        # update keyword arguments
        for key in self.base_env_kwargs:
            kwargs[key] = kwargs.get(key, self.base_env_kwargs[key])

        # validate arguments
        assert t_norm_max > t_norm_ssz, "maximum normalized time should be greater than the normalized step size"
        assert n_properties >= 0, "parameter ``n_properties`` should be non-negative"
        assert action_interval > 0, "parameter ``action_interval`` should be a positive integer"
        assert len(data_idxs) > 0, "parameter ``data_idxs`` should be a list containing at least one element"
        assert kwargs['observation_stds'] is None or type(kwargs['observation_stds']) is list, "parameter ``observation_stds`` should be a list"
        assert kwargs['seed'] is None or type(kwargs['seed']) is int, "parameter ``seed`` should be an integer or ``None``"
        assert type(kwargs['cache_all_data']) is bool, "parameter ``cache_all_data`` should be a boolean"
        assert kwargs['plot_interval'] > 0, "parameter ``plot_interval`` should be a positive integer"
        assert len(kwargs['plot_idxs']) == len(kwargs['axes_args']), "number of indices for plot should match number of axes arguments"
        assert len(kwargs['observation_space_range']) == 2, "parameter ``observation_space_range`` should contain two elements for the minimum and maximum values, both inclusive"
        assert len(kwargs['action_space_range']) == 2, "parameter ``action_space_range`` should contain two elements for the minimum and maximum values, both inclusive"
        assert kwargs['action_space_type'] in ['binary', 'box'], "parameter ``action_space_type`` can be either ``'binary'`` or ``'box'``"
        assert kwargs['cache_dump_interval'] > 0, "parameter ``cache_dump_interval`` should be a positive integer"

        # set backend
        self.backend = backend

        # frequently used variables
        self.numpy_int = self.backend.dtypes['numpy'][self.backend.precision]['integer']
        self.numpy_real = self.backend.dtypes['numpy'][self.backend.precision]['real']

        # time attributes
        self.t_norm_max = t_norm_max
        self.t_norm_ssz = t_norm_ssz
        self.t_norm_mul = t_norm_mul
        # truncate before maximum time if not divisible
        self.shape_T = (self.numpy_int(self.t_norm_max / self.t_norm_ssz) + 1, )
        self.T_norm = np.arange(self.shape_T[0], dtype=self.numpy_real) * self.t_norm_ssz
        self.T = self.backend.convert_to_typed(
            tensor=self.T_norm,
            dtype='real'
        ) * t_norm_mul
        self.t_ssz = self.t_norm_ssz * t_norm_mul

        # observation attributes
        self.n_observations = n_observations
        self.observation_space_range = kwargs['observation_space_range']
        self.observation_space = Box(
            low=self.observation_space_range[0],
            high=self.observation_space_range[1],
            shape=(self.n_observations, ),
            dtype=self.numpy_real
        )
        self.observation_stds = kwargs['observation_stds']
        if self.observation_stds is not None:
            self.observation_stds = self.backend.convert_to_typed(
                tensor=self.observation_stds,
                dtype='real'
            )

        # property attributes
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
                dtype=self.numpy_real
            )
        self.action_maximums = self.backend.convert_to_typed(
            tensor=action_maximums,
            dtype='integer' if self.action_space_type == 'binary' else 'real'
        )
        self.action_interval = action_interval
        self.action_steps = self.numpy_int(np.ceil((self.shape_T[0] - 1) / self.action_interval))

        # align delay with action interval
        self.has_delay = kwargs['has_delay']
        self.t_delay = self.T[self.action_interval] - self.T[0]

        # initialize seed
        self.seed = kwargs['seed']

        # data constants
        self.dir_path = dir_prefix + '/' + '_'.join([
            str(t_norm_max),
            str(t_norm_ssz),
            str(t_norm_mul),
            str(action_maximums),
            str(action_interval)
        ])
        self.file_path_prefix = self.dir_path + '/' + file_prefix
        self.n_data = 1 + self.n_actions + self.n_observations + self.n_properties + 1
        self.average_over = self.numpy_int(kwargs['average_over'])

        # initialize IO
        self.data_idxs = data_idxs
        self.cache_all_data = kwargs['cache_all_data']
        self.cache_dump_interval = kwargs['cache_dump_interval']
        self.io = FileIO(
            disk_cache_dir=self.file_path_prefix + '_cache',
            cache_dump_interval=self.cache_dump_interval
        )

        # plot constants
        self.plot = kwargs['plot']
        self.plot_interval = kwargs['plot_interval']
        self.plot_idxs = kwargs['plot_idxs']
        self.plot_buffer = kwargs['plot_buffer']
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
        self.T_step = None
        self.States = None
        self.Observations = None
        self.Reward = None
        self.Properties = None

    @abstractmethod
    def _update_states(self):
        """Method to update the states for each step.

        Returns
        -------
        States: Any
            The updated states with shape either ``(action_interval + 1, n_observations)`` or ``(action_interval + 1, n_envs, n_observations).
        """

        raise NotImplementedError

    @abstractmethod
    def reset_states(self):
        """Method to obtain the initial states.

        Returns
        -------
        states_0: Any
            The initial states with shape either ``(n_observations, )`` or ``(n_envs, n_observations)``, which are assigned to all elements of ``Observations`` with shape ``(action_interval + 1, n_observations, )`` or ``(action_interval + 1, n_envs, n_observations)``.
        """

        raise NotImplementedError

    def get_properties(self):
        """Method to obtain the properties calculated at each step.

        Returns
        -------
        Properties: Any
            The properties calculated from ``Observations`` with shape either ``(action_interval + 1, n_properties)`` or ``(action_interval + 1, n_envs, n_properties)``.
        """

        raise NotImplementedError

    @abstractmethod
    def get_reward(self):
        """Method to obtain the reward calculated at each step.

        Returns
        -------
        Reward: Any
            The reward calculated using ``States``, ``Observations`` or ``Properties`` with shape either ``(action_interval + 1, )`` or ``(action_interval + 1, n_envs)``.
        """

        raise NotImplementedError

    def validate_environment(self,
        shape_reset_states:tuple,
        shape_get_properties:tuple,
        shape_get_reward:tuple
    ):
        """Method to validate the interfaced environment.

        Parameters
        ----------
        shape_reset_states: tuple
            Shape of the array returned by ``reset_states``.
        shape_get_properties: tuple
            Shape of the array returned by ``get_properties``.
        shape_get_reward: tuple
            Shape of the array returned by ``get_reward``.
        """

        try:
            # validate initial states
            states_0 = self.backend.convert_to_typed(
                tensor=self.reset_states()
            )
            assert self.backend.shape(
                tensor=states_0
            ) == shape_reset_states, "``reset_states`` should return an array with shape ``{}``".format(shape_reset_states)
            # initialize states
            self.States = self.backend.repeat(
                tensor=self.backend.reshape(
                    tensor=states_0,
                    shape=(1, *shape_reset_states)
                ),
                repeats=self.action_interval + 1,
                axis=0
            )
            # initialize observations
            self.Observations = self.backend.update(
                tensor=self.Observations,
                indices=(slice(None), ),
                values=self.States
            )
            # validate properties
            if self.n_properties > 0:
                self.Properties = self.backend.convert_to_typed(
                    tensor=self.get_properties()
                )
                assert self.backend.shape(
                    tensor=self.Properties
                ) == shape_get_properties, "``get_properties`` should return an array with shape ``{}``".format(shape_get_properties)
            # validate reward
            self.Reward = self.backend.convert_to_typed(
                tensor=self.get_reward()
            )
            assert self.backend.shape(
                tensor=self.Reward
            ) == shape_get_reward, "``get_reward`` should return an array with shape ``{}``".format(shape_get_reward)
        except AttributeError as error:
            print(f"Missing required method or attribute: ({error}). Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for the implementation format of the missing method or add the missing attribute to the ``reset_states`` method.")
            exit()

    def reset(self):
        """Method to reset the time and obtain initial states as a typed tensor.

        Returns
        -------
        states_0: Any
            Inititial states.
        """

        # reset time
        self.t_idx = 0
        self.t = self.numpy_real(0.0)
        self.action_idx = 0

        # initialize states
        states_0 = self.backend.convert_to_typed(
            tensor=self.reset_states()
        )
        _shape = self.backend.shape(
            tensor=states_0
        )
        self.States = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=states_0,
                shape=(1, *_shape)
            ),
            repeats=self.action_interval + 1,
            axis=0
        )
        # initialize measurement noises
        if self.observation_stds is not None:
            self.Observation_noises = self.backend.normal(
                generator=self.backend.generator(
                    seed=self.seed
                ),
                shape=(self.shape_T[0], *_shape),
                mean=0.0,
                std=1.0,
                dtype='real'
            ) * self.backend.repeat(
                tensor=self.backend.reshape(
                    tensor=self.observation_stds,
                    shape=(1, *_shape)
                ),
                repeats=self.shape_T[0],
                axis=0
            )
        else:
            self.Observation_noises = self.backend.zeros(
                shape=(self.shape_T[0], *_shape),
                dtype='real'
            )
        # initialize observations
        observations_0 = states_0 + self.Observation_noises[0]
        self.Observations = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=observations_0,
                shape=(1, *self.backend.shape(
                    tensor=observations_0
                ))
            ),
            repeats=self.action_interval + 1,
            axis=0
        )

        return observations_0

    def update(self):
        """Method to update the time, observations, properties and reward and obtain the final set of observations and reward as typed tensors.

        Returns
        -------
        observations: Any
            Observations at current time.
        reward: Any
            Reward calculated.
        terminated: bool
            Flag to terminate trajectory.
        """

        # set evaluation times
        _dim_T = self.shape_T[0]
        self.T_step = self.T[self.t_idx:self.t_idx + self.action_interval + 1] if self.t_idx + self.action_interval < _dim_T else self.T[self.t_idx:]
        _dim_T_step = self.backend.shape(
            tensor=self.T_step
        )[0]

        # update actual states and observed states
        self.States = self._update_states()
        self.Observations = self.States + self.Observation_noises[self.t_idx:self.t_idx + _dim_T_step]

        # update properties
        if self.n_properties > 0:
            self.Properties = self.backend.convert_to_typed(
                tensor=self.get_properties()
            )

        # update rewards
        self.Reward = self.backend.convert_to_typed(
            tensor=self.get_reward()
        )

        # update time
        self.t_idx += _dim_T_step - 1
        self.t = self.T[self.t_idx]
        self.action_idx += 1

        # check if completed
        terminated = False if self.t_idx + 1 < _dim_T else True

        return self.Observations[_dim_T_step - 1], self.Reward[_dim_T_step - 1], terminated

    def check_truncation(self):
        """Method to check if the current episode needs to be truncated.

        Returns
        -------
        truncated: bool
            Whether to truncate the episode.
        """

        # check if out of bounds
        return bool(self.backend.max(
            tensor=self.Observations
        ) > self.observation_space_range[1] or self.backend.min(
            tensor=self.Observations
        ) < self.observation_space_range[0])

    def plot_learning_curve(self,
        data_rewards:np.ndarray=None,
        n_episodes:int=None,
        axis_args:list=None,
        hold:bool=False
    ):
        """Method to plot the learning curve.

        Either one of the parameters ``n_episodes`` or ``data_rewards`` should be provided.

        Parameters
        ----------
        data_rewards: :class:`numpy.ndarray`, default=None
            Cummulative rewards with shape ``(n_trajectories, 1)``. Loads data from disk cache if ``None``.
        n_episodes: int, default=None
            Total number of episodes to load from cache.
        axis_args: list, default=None
            Axis properties. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
        hold: bool, default=False
            Option to hold the plot.
        """

        # validate arguments
        assert data_rewards is not None or n_episodes is not None, "either one of the parameters ``n_episodes`` or ``data_rewards`` should be provided"

        # extract frequently used variables
        _idx_s = 0
        _idx_e = data_rewards.shape[0] - 1 if data_rewards is not None else n_episodes - 1
        file_name = self.file_path_prefix + '_' + '_'.join([
            'learning_curve',
            str(_idx_s),
            str(_idx_e)
        ])

        # get reward data from file
        if data_rewards is None:
            data_rewards = self.io.load_data(
                file_name=file_name
            )
        # get reward data from trajectories
        if data_rewards is None:
            data_rewards = self.io.get_disk_cache(
                idx_start=_idx_s,
                idx_end=_idx_e,
                idxs=[-1]
            )[:, -1, 0]

        # initialize plotter
        plotter = LearningCurvePlotter(
            axis_args=axis_args if axis_args is not None and len(axis_args) == 4 else self.default_axis_args_learning_curve,
            average_over=self.average_over
        )
        # update plot
        plotter.add_data(
            data_rewards=data_rewards,
            renew=False
        )
        # save plot
        self.io.save_data(
            data=data_rewards,
            file_name=file_name
        )
        plotter.save_plot(
            file_name=file_name
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
        make_gif:bool=True,
        hold:bool=False
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
        if hold:
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
        del self.T, self.T_norm, self.T_step, self.States, self.Observations, self.Reward
        if self.n_properties > 0:
            del self.Properties
        del self

class BaseGymEnv(BaseEnv, Env):
    r"""Gymnasium-based environment for reinforcement-learning.

    Refer to :class:`quantrl.envs.base.BaseEnv` for its documentation.
    """

    def __init__(self,
        backend:BaseBackend,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        data_idxs:list,
        dir_prefix:str,
        file_prefix:str,
        **kwargs
    ):
        """Class constructor for BaseGymEnv."""

        # initialize BaseEnv
        super().__init__(
            backend=backend,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            data_idxs=data_idxs,
            dir_prefix=dir_prefix,
            file_prefix=file_prefix,
            **kwargs
        )

        # initialize Gymnasium environment
        Env.__init__(self)

        # initialize buffers
        self.traj_idx = -1
        self.actions = None
        self.States = self.backend.empty(
            shape=(self.action_interval + 1, self.n_observations),
            dtype='real'
        )
        self.Observations = self.backend.empty(
            shape=(self.action_interval + 1, self.n_observations),
            dtype='real'
        )
        self.Properties = self.backend.empty(
            shape=(self.action_interval + 1, self.n_properties),
            dtype='real'
        )
        self.Reward = self.backend.empty(
            shape=(self.action_interval + 1, ),
            dtype='real'
        )
        self.rewards = None
        self.data_rewards = list()
        self.all_data = None
        self.data = None

    def validate_environment(self):
        return super().validate_environment(
            shape_reset_states=(self.n_observations, ),
            shape_get_properties=(self.action_interval + 1, self.n_properties),
            shape_get_reward=(self.action_interval + 1, )
        )

    def reset(self,
        seed:float=None,
        options:dict=None
    ):
        """Method to reset all variables for a new trajectory and obtain the initial observations as a NumPy array or a typed tensor.

        Parameters
        ----------
        seed: float
            Seed value for the reset.
        options: dict
            Options for the reset.

        Returns
        -------
        observations_0: Any
            Inititial observations.
        info: str
            Information on the reset.
        """

        # update buffers
        self.traj_idx += 1
        self.rewards = 0.0
        self.all_data = np.zeros((self.shape_T[0], self.n_data), dtype=self.numpy_real)

        # store selected data
        self.data = np.zeros((self.shape_T[0], len(self.data_idxs)), dtype=self.numpy_real)

        # reset variables
        observations_0 = super().reset()

        return observations_0, {
            'traj_idx': self.traj_idx
        }

    def step(self,
        actions
    ):
        """Method to take one single step and obtain the observations and reward as NumPy arrays or typed tensors.

        Parameters
        ----------
        actions: Any
            Actions at current time.

        Returns
        -------
        observations: Any
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
        self.actions = self.backend.convert_to_typed(
            tensor=actions,
            dtype='real'
        ) * self.action_maximums

        # get observations, properties and reward
        observations, reward, terminated = super().update()

        # update data
        self.update_data()

        # check if truncation required
        truncated = self.check_truncation()
        print(f'Trajectory #{self.traj_idx} truncated') if truncated > 0 else True

        # if trajectory ends
        if terminated or truncated:
            # update cache
            self.io.update_cache(
                data=self.all_data if self.cache_all_data else self.data
            )
            # update episode reward
            self.data_rewards.append(self.rewards)
            # update plotter
            if self.plot and self.traj_idx % self.plot_interval == 0:
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.all_data[:, self.plot_idxs],
                    traj_idx=self.traj_idx,
                    update_buffer=self.plot_buffer
                )

        return observations, reward, terminated, truncated, {}

    def update_data(self):
        """Method to update the trajectory data for the step.

        The first element at each time contains the current time.
        The next ``n_actions`` elements contain the actions.
        The next ``n_observations`` elements contain the observations.
        The next ``n_properties`` elements contain the properties.
        The final element is the cummulative reward from the step.
        """

        # frequently used variables
        _dim = self.backend.shape(
            tensor=self.T_step
        )[0]

        # update rewards
        self.rewards += self.Reward[_dim - 1]

        # extract indices
        _slice = slice(self.t_idx - _dim + 1, self.t_idx + 1)
        # extract values
        _Ts_step = self.backend.reshape(
            tensor=self.T_step,
            shape=(_dim, 1)
        )
        _actions = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=self.actions,
                shape=(1, self.n_actions)
            ),
            repeats=_dim,
            axis=0
        )
        _Rewards = self.backend.repeat(
            tensor=self.backend.convert_to_typed(
                tensor=[[self.rewards]],
                dtype='real'
            ),
            repeats=_dim,
            axis=0
        )
        # concatenate values
        _tensors = (_Ts_step, _actions, self.Observations, self.Properties, _Rewards) if self.n_properties > 0 else (_Ts_step, _actions, self.Observations, _Rewards)
        _data_backend = self.backend.concatenate(
                tensors=_tensors,
                axis=1,
                out=None
            )
        _data = self.backend.convert_to_numpy(
            tensor=_data_backend
        )

        # update data
        self.all_data[_slice, :] = _data
        self.data[_slice, :] = _data[:, self.data_idxs]

        # clear cache
        del _data_backend, _tensors, _Ts_step, _actions, _Rewards

    def evolve(self,
        show_progress=True,
        close=True
    ):
        """Method to freely evolve the trajectory.

        Parameters
        ----------
        close: bool, default=True
            Option to close the environment.
        """

        # reset environment
        self.reset()

        # simulate environment
        for _ in tqdm(
            range(self.action_steps),
            desc="Evolving",
            leave=True,
            mininterval=0.5,
            disable=not show_progress
        ):
            # set actions
            self.actions = self.action_maximums

            # step
            super().update()

            # update data
            self.update_data()

            # check if truncation required
            truncated = self.check_truncation()
            if truncated:
                print("Trajectory truncated")
                break

        # udpate cache
        self.io.update_cache(
            data=self.all_data if self.cache_all_data else self.data
        )

        # update plot
        if self.plot and self.traj_idx % self.plot_interval == 0:
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=self.all_data[:, self.plot_idxs],
                traj_idx=self.traj_idx,
                update_buffer=self.plot_buffer
            )

        # close environment
        if close:
            self.close(
                save=False
            )

    def close(self,
        save=True,
        axis_args=None
    ):
        """Method to close the environment.

        Parameters
        ----------
        save: bool, default=True
            Option to save the learning curve and make a gif file from the plot buffer.
        axis_args: list, default=None
            Axis properties for the learning curve. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
        """

        if save:
            _data_rewards = self.backend.convert_to_numpy(
                tensor=self.data_rewards,
                dtype='real'
            )
            # save learning curve
            self.plot_learning_curve(
                data_rewards=_data_rewards.reshape((len(_data_rewards), 1)),
                n_episodes=None,
                axis_args=axis_args,
                hold=False
            )
        del self.rewards, self.data_rewards, self.all_data, self.data

        # close io
        self.io.close(
            dump_cache=True
        )

        # clean
        super().close(
            n_episodes=self.traj_idx,
            save_replay=save
        )

class BaseSB3Env(BaseEnv, VecEnv):
    r"""Stable-Baselines3-based vectorized environments for reinforcement-learning.

    Initializes ``action_maximums_batch``.

    Refer to :class:`quantrl.envs.base.BaseEnv` for its documentation.
    The additional parameter ``n_envs`` denotes the number of environments to run in parallel and overrides the ``cache_dump_interval`` parameter.
    """

    def __init__(self,
        backend:BaseBackend,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_envs:int,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        data_idxs:list,
        dir_prefix:str,
        file_prefix:str,
        **kwargs
    ):
        """Class constructor for BaseSB3Env."""

        # initialize BaseEnv
        super().__init__(
            backend=backend,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            data_idxs=data_idxs,
            dir_prefix=dir_prefix,
            file_prefix=file_prefix,
            cache_dump_interval=n_envs,
            **kwargs
        )

        # update attributes
        self.n_envs = n_envs
        self.render_mode = [None] * n_envs
        self.action_maximums_batch = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=self.action_maximums,
                shape=(1, self.n_actions)
            ),
            repeats=self.n_envs,
            axis=0
        )

        # initialize SB3 environment
        VecEnv.__init__(self,
            num_envs=self.n_envs,
            observation_space=self.observation_space,
            action_space=self.action_space
        )

        # initialize buffers
        self.batch_idx = -1
        self.actions = None
        self.States = self.backend.empty(
            shape=(self.action_interval + 1, self.n_envs, self.n_observations),
            dtype='real'
        )
        self.Observations = self.backend.empty(
            shape=(self.action_interval + 1, self.n_envs, self.n_observations),
            dtype='real'
        )
        self.Properties = self.backend.empty(
            shape=(self.action_interval + 1, self.n_envs, self.n_properties),
            dtype='real'
        )
        self.Reward = self.backend.empty(
            shape=(self.action_interval + 1, self.n_envs),
            dtype='real'
        )
        self.rewards = None
        self.data_rewards = list()
        self.data = None
        if self.plot:
            self.env_idx_arr = np.arange(self.n_envs, dtype=self.numpy_int)

    def validate_environment(self):
        return super().validate_environment(
            shape_reset_states=(self.n_envs, self.n_observations),
            shape_get_properties=(self.action_interval + 1, self.n_envs, self.n_properties),
            shape_get_reward=(self.action_interval + 1, self.n_envs)
        )

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
        """Method to reset all variables for a new batch and obtain the initial observations as a NumPy array or a typed tensor.

        Parameters
        ----------
        seed: float
            Seed value for the reset.
        options: dict
            Options for the reset.

        Returns
        -------
        observations_0: Any
            Inititial observations.
        """

        # reset variables
        observations_0 = self._reset()

        return observations_0

    def _reset(self):
        """Method to reset all variables for a new batch and obtain the initial observations as a typed tensor.

        Returns
        -------
        observations_0: Any
            Inititial observations.
        """

        # update buffers
        self.batch_idx += 1
        self.rewards = self.backend.zeros(
            shape=(self.n_envs, ),
            dtype='real'
        )

        # store selected data and plot data
        self.data = np.zeros((self.n_envs, self.shape_T[0], len(self.data_idxs)),  dtype=self.numpy_real)
        if self.plot:
            self.plotter_env_idxs = self.env_idx_arr[(self.batch_idx * self.n_envs + self.env_idx_arr) % self.plot_interval == 0]
            self.plotter_env_data = np.zeros((len(self.plotter_env_idxs), self.shape_T[0], len(self.plot_idxs)), dtype=self.numpy_real)

        return super().reset()

    def step_async(self,
        actions
    ):
        """Method to prepare for one single step.

        Parameters
        ----------
        actions: Any
            Actions at current time.
        """

        # set actions
        self.actions = self.backend.convert_to_typed(
            tensor=actions,
            dtype='real'
        ) * self.action_maximums_batch

    def step_wait(self):
        """Method to take one single step and wait for the result.

        Returns
        -------
        observations: Any
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

        # update data
        self.update_data()

        # check if truncation required
        truncated = self.check_truncation()
        print(f'Batch #{self.batch_idx} truncated') if truncated > 0 else True

        # if trajectory ends
        if terminated or truncated:
            # update plotter and io
            if self.plot:
                for _i in range(len(self.plotter_env_idxs)):
                    self.plotter.plot_lines(
                        xs=self.T_norm,
                        Y=self.plotter_env_data[_i, :, :],
                        traj_idx=self.batch_idx * self.n_envs + self.plotter_env_idxs[_i],
                        update_buffer=self.plot_buffer
                    )
            # update episode reward
            self.data_rewards.append(self.rewards)

            # reset variables
            observations = self._reset()

        return observations, reward, [terminated or truncated] * self.n_envs, [{}] * self.n_envs

    def update_data(self):
        """Method to update the batch data for the step.

        The first element at each time contains the batch of stepped time.
        The next ``n_actions`` elements contain the batch of actions.
        The next ``n_observations`` elements contain the batch of observations.
        The next ``n_properties`` elements contain the batch of properties.
        The final element is the batch of cummulative reward from the step.
        """

        # frequently used variables
        _dim = self.backend.shape(
            tensor=self.T_step
        )[0]

        # update rewards
        self.rewards += self.Reward[_dim - 1]

        # extract values
        _Ts_step = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=self.T_step,
                shape=(1, _dim, 1)
            ),
            repeats=self.n_envs,
            axis=0
        )
        _Observations = self.backend.transpose(
            tensor=self.Observations[:_dim],
            axis_0=1,
            axis_1=0
        )
        if self.n_properties > 0:
            _Properties = self.backend.transpose(
                tensor=self.Properties[:_dim],
                axis_0=1,
                axis_1=0
            )
        _actions = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=self.actions,
                shape=(self.n_envs, 1, self.n_actions)
            ),
            repeats=_dim,
            axis=1
        )
        _Rewards = self.backend.repeat(
            tensor=self.backend.reshape(
                tensor=self.rewards,
                shape=(self.n_envs, 1, 1)
            ),
            repeats=_dim,
            axis=1
        )
        # concatenate values
        _tensors = (_Ts_step, _actions, _Observations, _Properties, _Rewards) if self.n_properties > 0 else (_Ts_step, _actions, _Observations, _Rewards)
        _data_backend = self.backend.concatenate(
            tensors=_tensors,
            axis=2,
            out=None
        )
        _data = self.backend.convert_to_numpy(
            tensor=_data_backend
        )

        # dump part data to disk
        if self.cache_all_data:
            self.io.dump_part_async(
                data=_data,
                batch_idx=self.batch_idx,
                part_idx=self.action_idx - 1
            )

        # update selected data
        _idx_start = self.t_idx - _dim + 1
        _idx_stop = self.t_idx + 1
        self.data[:, _idx_start:_idx_stop, :] = _data[:, :, self.data_idxs]
        # update plot data
        if self.plot:
            self.plotter_env_data[:, _idx_start:_idx_stop, :] = _data[self.plotter_env_idxs][:, :, self.plot_idxs]

        # clear cache
        del _data_backend, _tensors, _Ts_step, _actions, _Observations, _Rewards
        if self.n_properties > 0:
            del _Properties

    def evolve(self,
        show_progress=True,
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

        # reset environment
        self.reset()

        # simulate environment
        for _ in tqdm(
            range(self.action_steps),
            desc="Evolving",
            leave=True,
            mininterval=0.5,
            disable=not show_progress
        ):
            # set actions
            self.actions = self.action_maximums_batch

            # udpate reward
            super().update()
            
            # update data
            self.update_data()

            # check if truncation required
            truncated = self.check_truncation()
            if truncated:
                print("Batch truncated")
                break

        # update plot
        if self.plot:
            for _i in tqdm(
                range(len(self.plotter_env_idxs)),
                desc="Plotting",
                leave=True,
                mininterval=0.5,
                disable=False
            ):
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.plotter_env_data[_i, :, :],
                    traj_idx=self.batch_idx * self.n_envs + self.plotter_env_idxs[_i],
                    update_buffer=self.plot_buffer
                )
            self.plotter.hold_plot()

        # update episode reward
        self.data_rewards.append(self.rewards)

        # close environment
        if close:
            self.close(
                save=save
            )

    def close(self,
        save=True,
        axis_args=None
    ):
        """Method to close the environment.

        Parameters
        ----------
        save: bool, default=True
            Option to save the learning curve and make a gif file from the plot buffer.
        axis_args: list, default=None
            Axis properties for the learning curve. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``.
        """

        # save learning curve
        if save:
            _data_rewards = self.backend.convert_to_numpy(
                tensor=self.data_rewards,
                dtype='real'
            )
            self.plot_learning_curve(
                data_rewards=_data_rewards.reshape((_data_rewards.shape[0] * self.n_envs, 1)),
                n_episodes=None,
                axis_args=axis_args,
                hold=False
            )

        # clear cache
        del self.rewards, self.data_rewards, self.data
        if self.plot:
            del self.env_idx_arr, self.plotter_env_idxs, self.plotter_env_data

        # close io
        self.io.close(
            dump_cache=True
        )

        # clean
        super().close(
            n_episodes=self.batch_idx,
            save_replay=save
        )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module with environments for reinforcement learning."""

__name__    = 'quantrl.envs.rl'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-01-15"

# dependencies
from tqdm import tqdm
import gymnasium
import gymnasium.spaces as spaces
import numpy as np

# quantrl modules
from ..io import BaseIO
from ..plotters import BaseTrajectoryPlotter
from ..solvers.differential import IVPSolver

class BaseRLEnv(gymnasium.Env):
    r"""Base environment for reinforcement-learning.

    Initializes ``action_space``, ``observation_space``, ``t_norms``, ``io`` and ``plotter``.
    The custom environment needs to implement ``reset_env``, ``get_properties`` and ``get_reward`` methods.
    For SciPy-based solvers, the ``func`` method must be implemented.

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
        Option to plot the trajectories using ``:class:BaseEnvPlotter``. Default is ``True``.
    plot_interval: int, optional
        Number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
    plot_idxs: list, optional
        Positions of the data array to plot at each time step. Default is ``[-1]`` for the cummulative reward.
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
    """

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
        """Class constructor for BaseRLEnv."""

        # validate
        assert t_norm_max > t_norm_ssz, 'maximum normalized time should be greater than the normalized step size'
        assert action_interval > 0, 'parameter ``action_interval`` should be a positive integer'
        assert plot_interval >= 0, 'parameter ``plot_interval`` should be a non-negative integer'
        assert plot_interval < n_trajectories, 'parameter ``plot_interval`` should be a less than parameter ``n_trajectories``'
        assert len(plot_idxs) == len(axes_args), 'number of indices for plot should match number of axes arguments'
        
        # trajectory constants
        self.n_trajectories = n_trajectories
        # time constants
        self.t_norm_max = t_norm_max
        self.t_norm_ssz = t_norm_ssz
        self.t_dim = int(t_norm_max / t_norm_ssz) + 1
        self.t_norm_mul = t_norm_mul
        self.T_norm = np.linspace(0.0, t_norm_max, self.t_dim, dtype=np.float_)
        self.T = self.T_norm * t_norm_mul
        # step constants
        self.action_maxs = np.array(action_maxs, dtype=np.float_)
        self.action_interval = action_interval
        self.total_timesteps = self.n_trajectories * (self.t_dim - 1) / self.action_interval
        self.has_delay = has_delay
        self.t_delay = self.T[self.action_interval] - self.T[0]
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
        self.action_space = spaces.Box(
            low=action_space_range[0],
            high=action_space_range[1],
            shape=self.action_maxs.shape,
            dtype=np.float_
        )
        self.observation_space = spaces.Box(
            low=observation_space_range[0],
            high=observation_space_range[1],
            shape=observation_space_shape,
            dtype=np.float_
        )
        self.n_actions = self.action_space.shape[0]
        self.n_observations = self.observation_space.shape[0]

        # initialize IO
        self.io = BaseIO(
            disk_cache_dir=self.file_prefix + '_cache',
            max_cache_size=max_trajectories_per_file
        )

        # initialize plotter
        if self.plot:
            self.plotter = BaseTrajectoryPlotter(
                axes_args=axes_args,
                axes_lines_max=axes_lines_max,
                axes_cols=axes_cols,
                show_title=True
            )

        # buffer variables
        self.traj_idx = -1
        self.actions = np.zeros(self.n_actions, dtype=np.float_)
        self.Observations = np.zeros((self.action_interval + 1, self.n_observations), dtype=np.float_)
        # validate environment
        self.reset_Observations()
        self.Observations[1:] = self.Observations[0]
        self.n_data_elements = self.n_actions + self.n_observations + 2
        if self.save_properties:
            n_properties = np.shape(self.get_Properties())[1]
            self.n_data_elements += n_properties
        self.R = self.get_Reward()[-1]

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
        observations: :class:numpy.ndarray
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
                Y=self.data,
                idxs=self.plot_idxs,
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
        actions: :class:numpy.ndarray
            Array of actions at current time.
        
        Returns
        -------
        observations: :class:numpy.ndarray
            Array of observations at current time.
        reward: :class:numpy.float64
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
        # truncated = np.max(self.Observations) > self.observation_space_range[1] or np.min(self.Observations) < self.observation_space_range[0]
        truncated=False

        # check if completed
        terminated = False if self.t_idx + 1 < self.t_dim else True
        if terminated or truncated:
            print('truncated') if truncated else True
            if self.plot and self.plot_interval and self.traj_idx % self.plot_interval == 0:
                self.plotter.plot_lines(
                    xs=self.T_norm,
                    Y=self.data,
                    idxs=self.plot_idxs,
                    traj_idx=self.traj_idx
                )
            self.io.update_cache(self.data)

        return self.Observations[-1], np.float_(self.R - prev_reward), terminated, truncated, {}
    
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
                y_js=self.data[_idxs],
                j=_idxs[-1],
                idxs=self.plot_idxs
            )

    def evolve(self):
        """Method to freely evolve the trajectory."""

        # update actions
        self.actions = self.action_maxs

        for _ in tqdm(
            range(self.action_interval, self.t_dim, self.action_interval),
            desc='Progress (time)',
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
                Y=self.data,
                idxs=self.plot_idxs
            )

    def close(self):
        """Method to close the environment."""

        self.io.close()
    
    def replay_trajectories(self,
        idx_start:int=0,
        idx_end:int=None
    ):
        """Method to replay trajectories in a given range.
        
        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        """

        # get trajectory data in the given range
        all_data = self.io.get_disk_cache(
            idx_start=idx_start,
            idx_end=(self.n_trajectories - 1) if idx_end is None else idx_end
        )
        # play animation
        _interval = self.plot_interval if self.plot_interval != 0 else 1
        for i, Y in enumerate(all_data[::_interval]):
            self.plotter.plot_lines(
                xs=self.T_norm,
                Y=Y,
                idxs=self.plot_idxs,
                traj_idx=idx_start + i * _interval,
        )
        # hold plot
        self.plotter.show_plot()

    def plot_learning_curve(self,
        axis_args=['$N$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']
    ):
        """Method to plot the learning curve.

        Parameters
        ----------
        axis_args: list
            Axis properties. The first element is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``['$N$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']``.
        """

        # get all trajectory data
        all_data = self.io.get_disk_cache(
            idx_start=0,
            idx_end=self.n_trajectories - 1
        )
        # reinitialize plotter
        plotter = BaseTrajectoryPlotter(
            axes_args=[axis_args],
            axes_lines_max=1,
            axes_cols=1,
            show_title=False
        )
        # plot
        plotter.plot_lines(
            xs=list(range(len(all_data))),
            Y=np.squeeze(all_data[:, -1, :]),
            idxs=[-1]
        )
        # hold
        plotter.show_plot()

class LinearizedHORLEnv(BaseRLEnv):
    """Class to interface linearized harmonic oscillator environments for reinforcement learning.

    Initializes ``dim_corrs``, ``num_corrs``, ``is_A_constant`` and ``is_D_constant``.
    
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.

    Parameters
    ----------
    params: dict
        Parameters of the environment.
    name: str
        Name of the environment.
    num_modes: int
        Number of classical mode amplitudes in the environment.
    num_quads: int
        Number of quantum fluctuation quadratures in the environment.
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
    solver_type: str, optional
        Solver to evolve each time step. Options are ``'qtraj'`` for quantum trajectories using the ``get_H_eff(t)`` method, ``'wiener'`` for Wiener process using the ``get_M(t)`` method and ``'scipy'`` for SciPy-based ODE/DDE solvers using ``:class:quantrl.solvers.differential.IVPSolver``. Default is ``'scipy'``.
    solver_method: str, optional
        Method used to solve the ODEs/DDEs. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``. Default is ``'vode'``.
    solver_atol: float, optional
        Absolute tolerance of the SciPy-based ODE/DDE solver. Default is ``1e-12``.
    solver_rtol: float, optional
        Relative tolerance of the SciPy-based ODE/DDE solver. Default is ``1e-6``.
    plot: bool, optional
        Option to plot the trajectories using ``:class:BaseEnvPlotter``. Default is ``True``.
    plot_interval: int, optional
        Number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
    plot_idxs: list, optional
        Positions of the data array to plot at each time step. Default is ``[-1]`` for the cummulative reward.
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
    """

    # attribute
    default_params = dict()

    def __init__(self,
        params:dict,
        name:str,
        desc:str,
        num_modes:int,
        num_quads:int,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        action_maxs:list,
        action_interval:int=1,
        has_delay:bool=False,
        reward_max:float=1.0,
        reward_noise:float=0.0,
        action_space_range:list=[-1, 1],
        observation_space_range:list=[-1e6, 1e6],
        observation_space_shape:tuple=(10, ),
        save_properties:bool=False,
        solver_type:str='scipy',
        solver_method:str='vode',
        solver_atol:float=1e-12,
        solver_rtol:float=1e-6,
        plot:bool=True,
        plot_interval:int=1,
        plot_idxs:list=[-1],
        axes_args:list=[
            ['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']
        ],
        axes_lines_max:int=100,
        axes_cols:int=3,
        dir_prefix:str='data/env',
        max_trajectories_per_file:int=100
    ):
        """Class constructor for LinearizedHORLEnv."""

        # set constants
        self.name = name
        self.desc = desc
        self.params = params
        self.num_modes = num_modes
        self.dim_corrs = (num_quads, num_quads)
        self.num_corrs = num_quads**2
        self.is_A_constant = False
        self.is_D_constant = False

        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # initialize base environment
        super().__init__(
            n_trajectories=n_trajectories,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            action_maxs=action_maxs,
            action_interval=action_interval,
            has_delay=has_delay,
            reward_max=reward_max,
            reward_noise=reward_noise,
            action_space_range=action_space_range,
            observation_space_range=observation_space_range,
            observation_space_shape=observation_space_shape,
            save_properties=save_properties,
            plot=plot,
            plot_interval=plot_interval,
            plot_idxs=plot_idxs,
            axes_args=axes_args,
            axes_lines_max=axes_lines_max,
            axes_cols=axes_cols,
            dir_prefix=(dir_prefix if dir_prefix != 'data/env' else ('data/' + self.name.lower())) + '_' + '_'.join([
                str(self.params[key]) for key in self.params
            ]),
            max_trajectories_per_file=max_trajectories_per_file
        )

        # initialize solver
        self.solver_type = solver_type
        if 'wiener' in self.solver_type:
            self.W_is = np.sqrt(self.t_norm_ssz) * np.random.normal(loc=0.0, scale=1.0, size=(n_trajectories, self.t_dim))
        else:
            self.solver = IVPSolver(
                func=self.func,
                solver_params={
                    't_min': self.T[0],
                    't_max': self.T[-1],
                    't_dim': self.t_dim,
                    'ode_method': solver_method,
                    'ode_atol': solver_atol,
                    'ode_rtol': solver_rtol,
                    'ode_is_stiff': False,
                    'ode_step_dim': self.action_interval
                },
                func_controls=getattr(self, 'func_controls', None),
                has_delay=self.has_delay,
                func_delay=getattr(self, 'func_delay', None),
                t_delay=self.t_delay
            )

        # initialize buffer variables
        if self.num_modes != 0:
            self.mode_rates_real = np.zeros(2 * self.num_modes, dtype=np.float_)
        if self.num_corrs != 0:
            self.A = np.zeros(self.dim_corrs, dtype=np.float_)
            self.D = np.zeros(self.dim_corrs, dtype=np.float_)
            self.matmul_0 = np.empty(self.dim_corrs, dtype=np.float_)
            self.matmul_1 = np.empty(self.dim_corrs, dtype=np.float_)
            self.sum_0 = np.empty(self.dim_corrs, dtype=np.float_)
            self.sum_1 = np.empty(self.dim_corrs, dtype=np.float_)
        self.y_rates = np.empty(2 * self.num_modes + self.num_corrs, dtype=np.float_)

    def _step(self,
        actions
    ):
        """Method to implement one step.

        Parameters
        ----------
        actions: :class:numpy.ndarray
            Array of actions at current time.
        """

        # Wiener processes
        if self.solver_type == 'wiener':
            return self._step_wiener(
                actions=actions
            )
        # coupled ODEs/DDEs
        else:
            return self.solver.step_ivp(
                y0=self.Observations[-1],
                T_step=self.T_step,
                params=actions
            )
    
    def _step_wiener(self):
        """Method to implement one step of a Wiener process.

        Parameters
        ----------
        actions: :class:numpy.ndarray
            Array of actions at current time.

        Returns
        -------
        Observations: :class:numpy.ndarray
            Observations for the action interval.
        """

        # TODO: vectorize
        Observations = np.zeros((self.action_interval, self.n_observations), dtype=np.float_)
        Observations[0] = self.Observations[-1]
        for i in range(1, self.action_interval + 1):
            M_i = self.get_M_i(self.t_idx + i)
            n_i = self.get_noise_prefix() * self.W_is[self.traj_idx, self.t_idx + i]
            Observations[i] = M_i.dot(Observations[i - 1]) + n_i

        return Observations
    
    def func(self,
        t,
        y,
        args
    ):
        r"""Wrapper function for the rates of change of the real-valued modes and correlations. 

        Requires system method ``get_mode_rates``. Additionally, ``get_A`` should be defined for the correlations. Optionally, ``get_D`` may be defined for correlations. Refer to **Notes** for their implementations.
        
        The variables are casted to real-valued.
        
        Parameters
        ----------
        t : float
            Time at which the values are calculated.
        y : numpy.ndarray
            Real-valued modes and flattened correlations. First ``num_modes`` elements contain the real parts of the modes, the next ``num_modes`` elements contain the imaginary parts of the modes and optionally, the last ``num_corrs`` elements contain the correlations.
        args : tuple
            Delay function, control function and constant parameters.

        Returns
        -------
        rates : numpy.ndarray
            Rates of change of the real-valued modes and flattened correlations.
        """

        # extract frequently used variables
        if self.num_modes != 0:
            modes = y[:self.num_modes] + 1.0j * y[self.num_modes:2 * self.num_modes]
            # get real-valued mode rates
            self.y_rates[:2 * self.num_modes] = self.get_mode_rates_real(
                t=t,
                modes_real=y[:2 * self.num_modes],
                args=args
            )
        else:
            modes=None

        if self.num_corrs != 0:
            corrs = np.reshape(y[2 * self.num_modes:], self.dim_corrs)

            # get drift matrix
            self.A = self.A if self.is_A_constant else self.get_A(
                t=t,
                modes=modes,
                args=args
            )

            # get noise matrix
            self.D = self.D if self.is_D_constant else self.get_D(
                t=t,
                modes=modes,
                corrs=corrs,
                args=args
            )
            # get flattened correlation rates
            self.y_rates[2 * self.num_modes:] = np.add(np.add(np.matmul(self.A, corrs, out=self.matmul_0), np.matmul(corrs, self.A.transpose(), out=self.matmul_1), out=self.sum_0), self.D, out=self.sum_1).ravel()

        return self.y_rates

    def get_mode_rates_real(self, t, modes_real, args):
        """Method to obtain the real-valued mode rates from real-valued modes.

        Requires the system method ``get_mode_rates``. Refer to **Notes** for its implementation.

        Parameters
        ----------
        t : float
            Time at which the drift matrix is calculated.
        modes_real : numpy.ndarray
            Real-valued modes.
        args : tuple
            Delay function, control function and constant parameters.
        
        Returns
        -------
        mode_rates_real : numpy.ndarray
            Real-valued rates for each mode.
        """

        # handle null
        if getattr(self, 'get_mode_rates', None) is None:
            return self.mode_rates_real

        # get complex-valued mode rates
        mode_rates = self.get_mode_rates(
            t=t,
            modes=modes_real[:self.num_modes] + 1.0j * modes_real[self.num_modes:],
            args=args
        )

        # set real-valued mode rates
        self.mode_rates_real[:self.num_modes] = np.real(mode_rates)
        self.mode_rates_real[self.num_modes:] = np.imag(mode_rates)

        return self.mode_rates_real

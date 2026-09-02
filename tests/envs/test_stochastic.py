"""Module to test `quantrl.envs.stochastic`"""

# dependencies
import numpy as np
import pytest

# quantrl modules
from quantrl.envs.stochastic import LinearEnv

class SHOEnv00(LinearEnv):
    """Class to simulate a simple harmonic oscillator"""
    default_params = {
        'n_th': 1e4,
    }
    def __init__(self,
            backend_library='numpy',
            seed=None,
        ):
        # initialize Gym environment
        super().__init__(
            name='SHOEnv',
            desc="Simple Harmonic Oscillator Environment",
            params={},
            t_norm_max=10.0,
            t_norm_ssz=0.001,
            t_norm_mul=2.0 * np.pi,
            n_observations=2,
            n_properties=0,
            n_actions=1,
            action_maximums=[0.0],
            action_interval=100,
            data_idxs=[2, 3],
            observation_stds=[0.1] * 2,
            backend_library=backend_library,
            action_space_range=[-1.0, 1.0],
            observation_space_range=[-1e12, 1e12],
            seed=seed,
            cache_dump_interval=100,
            average_over=100,
            plot=False,
        )

        # set parameters
        self.Omega_norm = 1.0
        self.n_th = self.params['n_th']

        # update drift matrix
        self.A = self.backend.update(
            tensor=self.A,
            indices=(
                [0, 1],
                [1, 0],
            ),
            values=self.backend.convert_to_typed(
                tensor=[
                    self.Omega_norm,
                    - self.Omega_norm
                ],
                dtype='real',
            )
        )

        # set noise prefixes
        self.noise_prefixes = self.backend.zeros(
            shape=(2, ),
            dtype='real'
        )

    def reset_states(self):
        # set initial values of position and momentum
        # with mean 0 and standard deviation n_th + 0.5
        states_0 = self.backend.convert_to_typed(
            tensor=[-58.40454235, 256.72449014],
            dtype='real'
        )

        return states_0

    def get_A(self, t_idx, args):
        # update drift matrix
        return self.A

    def get_noise_prefixes(self, t_idx, args):
        # return noise prefixes
        return self.noise_prefixes

    def get_reward(self):
        # thermal occupancies
        ns = 0.5 * (self.Observations[:, 0]**2 + self.Observations[:, 1]**2)
        self.Reward = 1.0 / ns

        return self.Reward

@pytest.mark.parametrize(
    'backend_library',
    ['jax', 'torch', 'numpy'],
)
def test_sho_env(
        backend_library,
    ):
    """Function to test evolution."""
    env = SHOEnv00(backend_library=backend_library, seed=1234)
    env.evolve(show_progress=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to solve for various classical and quantum signatures and system measures.

References
----------

.. [1] R. Simon, *Peres-Horodechi Separability Criterion for Continuous Variable Systems*, Phys. Rev. Lett. **84**, 2726 (2000).

.. [2] D. Vitali, S. Gigan, A. Ferreira, H. R. Bohm, P. Tombesi, A. Guerreiro, V. Vedral, A. Zeilinger and M. Aspelmeyer, *Quantum Entanglement between a Movable Mirror and a Cavity Field*, Phys. Rev. Lett. **98**, 030405 (2007).

.. [3] S. Olivares, *Quantum Optics in Phase Space: A Tutorial on Gaussian States*, Eur. Phys. J. Special Topics **203**, 3 (2012).

.. [4] M. Ludwig and F. Marquardt, *Quantum Many-body Dynamics in Optomechanical Arrays*, Phys. Rev. Lett. **111**, 073603 (2013).

.. [5] A. Mari, A. Farace, N. Didier, V. Giovannetti and R. Fazio, *Measures of Quantum Synchronization in Continuous Variable Systems*, Phys. Rev. Lett. **111**, 103605 (2013).

.. [6] F. Galve, G. L. Giorgi and R. Zambrini, *Quantum Correlations and Synchronization Measures*, Lectures on General Quantum Correlations and their Applications, Quantum Science and Technology, Springer (2017).

.. [7] N. Yang, A. Miranowicz, Y.-C. Liu, K. Xia and F. Nori, *Chaotic Synchronization of Two Optical Cavity Modes in Optomechanical Systems*, Sci. Rep. ***9***, 15874 (2019).
"""

__name__ = 'qom.solvers.measure'
__authors__ = ["Sampreet Kalita"]
__created__ = "2021-01-04"
__updated__ = "2024-10-14"

# dependencies
from typing import Union

import numpy as np

class QCMSolver():
    r"""Class to solve for quantum correlation measures.

    Initializes ``Modes``, ``Corrs``, ``Omega_s`` (symplectic matrix), ``params`` and ``updater``.

    Parameters
    ----------
    Modes : numpy.ndarray
        Classical modes with shape ``(dim, num_modes)``.
    Corrs : numpy.ndarray
        Quadrature quadrature correlations with shape ``(dim, 2 * num_modes, 2 * num_modes)``.
    params : dict
        Parameters for the solver. Available options are:

            ================    ====================================================
            key                 value
            ================    ====================================================
            'show_progress'     (*bool*) option to display the progress of the solver. Default is ``False``.
            'measure_codes'     (*list* or *str*) codenames of the measures to calculate. Options are ``'discord_G'`` for Gaussian quantum discord [3]_, ``'entan_ln'`` for quantum entanglement (using matrix multiplications, fallback) [1]_, ``'entan_ln_2'`` for quantum entanglement (using analytical expressions) [2]_, ``'sync_c'`` for complete quantum synchronization [5]_, ``'sync_p'`` for quantum phase synchronization [5]_). Default is ``['entan_ln']``.
            'indices'           (*list* or *tuple*) indices of the modes as a list or tuple of two integers. Default is ``(0, 1)``.
            ================    ====================================================
    cb_update : callback, optional
        Callback function to update status and progress, formatted as ``cb_update(status, progress, reset)``, where ``status`` is a string, ``progress`` is a float and ``reset`` is a boolean.
    """

    # attributes
    name = 'QCMSolver'
    """str : Name of the solver."""
    desc = "Quantum Correlations Measure Solver"
    """str : Description of the solver."""
    method_codes = {
        'corrs_P_p': 'get_correlation_Pearson',
        'corrs_P_q': 'get_correlation_Pearson',
        'discord_G': 'get_discord_Gaussian',
        'entan_ln': 'get_entanglement_logarithmic_negativity',
        'entan_ln_2': 'get_entanglement_logarithmic_negativity_2',
        'sync_c': 'get_synchronization_complete',
        'sync_p': 'get_synchronization_phase'
    }
    """dict : Codenames of available methods."""
    solver_defaults = {
        'show_progress': False,
        'measure_codes': ['entan_ln'],
        'indices': (0, 1)
    }
    """dict : Default parameters of the solver."""

    def __init__(self, Modes, Corrs, params:dict, cb_update=None):
        """Class constructor for QCMSolver."""

        # validate modes and correlations
        self.Modes, self.Corrs = validate_Modes_Corrs(
            Modes=Modes,
            Corrs=Corrs
        )

        # set symplectic matrix
        self.Omega_s = np.kron(np.eye(2, dtype=np.float_), np.array([[0, 1], [-1, 0]], dtype=np.float_))

        # set parameters
        self.set_params(params)

        # set callback
        self.cb_update = cb_update

    def set_params(self, params:dict):
        """Method to validate and set the solver parameters.

        Parameters
        ----------
        params : dict
            Parameters of the solver.
        """

        # check required parameters
        assert 'measure_codes' in params, "Parameter ``params`` does not contain a required key ``'measure_codes'``"
        assert 'indices' in params, "Parameter ``params`` does not contain a required key ``'indices'``"

        # extract frequently used variables
        measure_codes = params['measure_codes']
        indices = params['indices']
        _dim = len(self.Modes[0]) if self.Modes is not None else int(len(self.Corrs[0]) / 2)

        # validate measure type
        assert isinstance(measure_codes, Union[list, str].__args__), "Value of key ``'measure_codes'`` can only be of types ``list`` or ``str``"
        # convert to list
        measure_codes = [measure_codes] if isinstance(measure_codes, str) else measure_codes
        # check elements
        for measure_code in measure_codes:
            assert measure_code in self.method_codes, f"Elements of key ``'measure_codes'`` can only be one or more keys of ``{self.method_codes.keys()}``"
        # update parameter
        params['measure_codes'] = measure_codes

        # validate indices
        assert isinstance(indices, Union[list, tuple].__args__), "Value of key ``'indices'`` can only be of types ``list`` or ``tuple``"
        # convert to list
        indices = list(indices) if isinstance(indices, tuple) else indices
        # check length
        assert len(indices) == 2, "Value of key ``'indices'`` can only have 2 elements"
        assert indices[0] < _dim and indices[1] < _dim, f"Elements of key ``'indices'`` cannot exceed the total number of modes ({_dim})"
        # update parameter
        params['indices'] = indices

        # set solver parameters
        self.params = {}
        for key, _ in self.solver_defaults.items():
            self.params[key] = params.get(key, self.solver_defaults[key])

    def get_measures(self):
        """Method to obtain the each measure.

        Returns
        -------
        Measures : numpy.ndarray
            Measures calculated with shape ``(dim, num_measure_codes)``.
        """

        # extract frequently used variables
        show_progress = self.params['show_progress']
        measure_codes = self.params['measure_codes']
        indices = self.params['indices']
        _dim = (len(self.Corrs), len(self.params['measure_codes']))

        # initialize measures
        Measures = np.zeros(_dim, dtype=np.float_)

        # find measures
        for j in range(_dim[1]):
            # display progress
            if show_progress and self.cb_update is not None:
                self.cb_update(
                    status="-" * (35 - len(measure_codes[j])) + "Obtaining Measures (" + measure_codes[j] + ")",
                    progress=0.0,
                    reset=False
                )

            func_name = self.method_codes[measure_codes[j]]

            # calculate measure
            Measures[:, j]  = getattr(self, func_name)(pos_i=2 * indices[0], pos_j=2 * indices[1]) if 'corrs_P_p' not in measure_codes[j] else getattr(self, func_name)(pos_i=2 * indices[0] + 1, pos_j=2 * indices[1] + 1)

        # display completion
        if show_progress and self.cb_update is not None:
            self.cb_update(
                status="-" * 39 + "Measures Obtained",
                progress=1.0,
                reset=False
            )

        return Measures

    def get_submatrices(self, pos_i:int, pos_j:int):
        """Helper function to obtain the block matrices of the required modes and its components.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Corrs_modes : numpy.ndarray
            Correlation matrix of the required modes.
        A: numpy.ndarray
            Correlation matrix of the first mode.
        B: numpy.ndarray
            Correlation matrix of the first mode.
        C: numpy.ndarray
            Correlation matrix of the cross mode.
        """

        # correlation matrix of the ith mode
        As = self.Corrs[:, pos_i:pos_i + 2, pos_i:pos_i + 2]
        # correlation matrix of the jth mode
        Bs = self.Corrs[:, pos_j:pos_j + 2, pos_j:pos_j + 2]
        # correlation matrix of the intermodes
        Cs = self.Corrs[:, pos_i:pos_i + 2, pos_j:pos_j + 2]

        # get transposes matrices
        C_Ts = np.array(np.transpose(Cs, axes=(0, 2, 1)))

        # get block matrices
        Corrs_modes = np.concatenate((np.concatenate((As, Cs), axis=2), np.concatenate((C_Ts, Bs), axis=2)), axis=1)

        # # correlation matrix of the two modes (slow)
        # Corrs_modes = np.array([np.block([[As[i], Cs[i]], [C_Ts[i], Bs[i]]]) for i in range(len(self.Corrs))], dtype=np.float_)

        return Corrs_modes, As, Bs, Cs

    def get_invariants(self, pos_i:int, pos_j:int):
        """Helper function to calculate symplectic invariants for two modes given the correlation matrices of their quadratures.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int8
            Index of jth quadrature.

        Returns
        -------
        I_1s : numpy.ndarray
            Determinants of ``A``.
        I_2s : numpy.ndarray
            Determinants of ``B``.
        I_3s : numpy.ndarray
            Determinants of ``C``.
        I_4s : numpy.ndarray
            Determinants of ``corrs_modes``.
        """

        # get block matrices and its components
        Corrs_modes, As, Bs, Cs = self.get_submatrices(
            pos_i=pos_i,
            pos_j=pos_j
        )

        # symplectic invariants
        return np.linalg.det(As), np.linalg.det(Bs), np.linalg.det(Cs), np.linalg.det(Corrs_modes)

    def get_correlation_Pearson(self, pos_i:int, pos_j:int):
        r"""Method to obtain the Pearson correlation coefficient.

        The implementation measure reads as [6]_,

        .. math::

            C_{ij} = \frac{\Sigma_{t} \langle \delta \mathcal{O}_{i} (t) \delta \mathcal{O}_{j} (t) \rangle}{\sqrt{\Sigma_{t} \langle \delta \mathcal{O}_{i}^{2} (t) \rangle} \sqrt{\Sigma_{t} \langle \delta \mathcal{O}_{j}^{2} (t) \rangle}}

        where :math:`\delta \mathcal{O}_{i}` and :math:`\delta \mathcal{O}_{j}` are the corresponding quadratures of quantum fluctuations.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Corr_P : float
            Pearson correlation coefficients.
        """

        # extract mean values of correlation elements
        mean_ii = np.mean(self.Corrs[:, pos_i, pos_i])
        mean_ij = np.mean(self.Corrs[:, pos_i, pos_j])
        mean_jj = np.mean(self.Corrs[:, pos_j, pos_j])

        # Pearson correlation coefficient as a repeated array
        return np.array([mean_ij / np.sqrt(mean_ii * mean_jj)] * len(self.Corrs), dtype=np.float_)

    def get_discord_Gaussian(self, pos_i:int, pos_j:int):
        """Method to obtain Gaussian quantum discord values [3]_.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Discord_G : numpy.ndarray
            Gaussian quantum discord values.
        """

        # initialize values
        mu_pluses = np.zeros(len(self.Corrs), dtype=np.float_)
        mu_minuses = np.zeros(len(self.Corrs), dtype=np.float_)
        Ws = np.zeros(len(self.Corrs), dtype=np.float_)
        Discord_G = np.zeros(len(self.Corrs), dtype=np.float_)

        # get symplectic invariants
        I_1s, I_2s, I_3s, I_4s = self.get_invariants(
            pos_i=pos_i,
            pos_j=pos_j
        )

        # sum of symplectic invariants
        sigmas = I_1s + I_2s + 2 * I_3s
        # discriminants of the simplectic eigenvalues
        _discriminants = sigmas**2 - 4 * I_4s
        # check sqrt condition
        conditions_mu = np.logical_and(_discriminants >= 0.0, I_4s >= 0.0)
        # update valid symplectic eigenvalues
        mu_pluses[conditions_mu] = 1 / np.sqrt(2) * np.sqrt(sigmas[conditions_mu] + np.sqrt(_discriminants[conditions_mu]))
        mu_minuses[conditions_mu] = 1 / np.sqrt(2) * np.sqrt(sigmas[conditions_mu] - np.sqrt(_discriminants[conditions_mu]))

        # check main condition on W values
        conditions_W = 4 * (np.multiply(I_1s, I_2s) - I_4s)**2 / (I_1s + 4 * I_4s) / (1 + 4 * I_2s) / I_3s**2 <= 1.0
        # W values with main condition
        # check sqrt and NaN condition
        _discriminants = 4 * I_3s**2 + np.multiply(4 * I_2s - 1, 4 * I_4s - I_1s)
        _divisors = 4 * I_2s - 1
        conditions_W_1 = np.logical_and(conditions_W, np.logical_and(_discriminants >= 0.0, _divisors != 0.0))
        # update W values
        Ws[conditions_W_1] = ((2 * np.abs(I_3s[conditions_W_1]) + np.sqrt(_discriminants[conditions_W_1])) / _divisors[conditions_W_1])**2
        # W values without main condition
        # check sqrt and NaN condtition
        _bs = np.multiply(I_1s, I_2s) + I_4s - I_3s**2
        _4acs = 4 * np.multiply(np.multiply(I_1s, I_2s), I_4s)
        _discriminants = _bs**2 - _4acs
        conditions_W_2 = np.logical_and(np.logical_not(conditions_W), np.logical_and(_discriminants >= 0.0, I_2s != 0.0))
        # update W values
        Ws[conditions_W_2] = (_bs[conditions_W_2] - np.sqrt(_bs[conditions_W_2]**2 - _4acs[conditions_W_2])) / 2 / I_2s[conditions_W_2]

        # all validity conditions
        conditions = np.logical_and(conditions_mu, np.logical_or(conditions_W_1, conditions_W_2))

        # f function
        def func_f(x):
            return np.multiply(x + 0.5, np.log10(x + 0.5)) - np.multiply(x - 0.5, np.log10(x - 1 / 2))

        # update quantum discord values
        Discord_G[conditions] = func_f(np.sqrt(I_2s[conditions])) \
                                - func_f(mu_pluses[conditions]) \
                                - func_f(mu_minuses[conditions]) \
                                + func_f(np.sqrt(Ws[conditions]))

        return Discord_G

    def get_entanglement_logarithmic_negativity(self, pos_i:int, pos_j:int):
        """Method to obtain the logarithmic negativity entanglement values using matrices [1]_.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Entan_lns : numpy.ndarray
            Logarithmic negativity entanglement values using matrices.
        """

        # get correlation matrix and its components
        Corrs_modes, _, _, _  = self.get_submatrices(
            pos_i=pos_i,
            pos_j=pos_j
        )

        # PPT criteria
        Corrs_modes[:, :, -1] = - Corrs_modes[:, :, -1]
        Corrs_modes[:, -1, :] = - Corrs_modes[:, -1, :]

        # smallest symplectic eigenvalue
        eigs, _ = np.linalg.eig(np.matmul(self.Omega_s, Corrs_modes))
        eigs_min = np.min(np.abs(eigs), axis=1)

        # initialize entanglement
        Entan_ln = np.zeros_like(eigs_min, dtype=np.float_)

        # update entanglement
        for i, eig_min in enumerate(eigs_min):
            if eig_min < 0:
                Entan_ln[i] = 0
            else:
                Entan_ln[i] = np.maximum(0.0, - np.log(2 * eig_min))

        return Entan_ln

    def get_entanglement_logarithmic_negativity_2(self, pos_i:int, pos_j:int):
        """Method to obtain the logarithmic negativity entanglement values using analytical expression [2]_.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Entan_ln : numpy.ndarray
            Logarithmic negativity entanglement values using analytical expression.
        """

        # initialize values
        Entan_ln = np.zeros(len(self.Corrs), dtype=np.float_)

        # symplectic invariants
        I_1s, I_2s, I_3s, I_4s = self.get_invariants(
            pos_i=pos_i,
            pos_j=pos_j
        )

        # sum of symplectic invariants after positive partial transpose
        sigmas = I_1s + I_2s - 2 * I_3s
        # discriminants of the simplectic eigenvalues
        discriminants = sigmas**2 - 4 * I_4s

        # check positive sqrt values
        conditions = np.logical_and(discriminants >= 0.0, I_4s >= 0.0)

        # calculate enganglement for positive sqrt values
        Entan_ln[conditions] = - 1 * np.log(2 / np.sqrt(2) * np.sqrt(sigmas[conditions] - np.sqrt(discriminants[conditions])))

        # clip negative values
        Entan_ln[Entan_ln < 0.0] = 0.0

        return Entan_ln

    def get_synchronization_complete(self, pos_i:int, pos_j:int):
        """Method to obtain the complete quantum synchronization values [5]_.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Sync_c : numpy.ndarray
            Complete quantum synchronization values.
        """

        # square difference between position quadratures
        q_minus_2s = 0.5 * (self.Corrs[:, pos_i, pos_i] + self.Corrs[:, pos_j, pos_j] - 2 * self.Corrs[:, pos_i, pos_j])
        # square difference between momentum quadratures
        p_minus_2s = 0.5 * (self.Corrs[:, pos_i + 1, pos_i + 1] + self.Corrs[:, pos_j + 1, pos_j + 1] - 2 * self.Corrs[:, pos_i + 1, pos_j + 1])

        # complete quantum synchronization values
        return 1.0 / (q_minus_2s + p_minus_2s)

    def get_synchronization_phase(self, pos_i:int, pos_j:int):
        """Method to obtain the quantum phase synchronization values [5]_.

        Parameters
        ----------
        pos_i : int
            Index of ith quadrature.
        pos_j : int
            Index of jth quadrature.

        Returns
        -------
        Sync_p : numpy.ndarray
            Quantum phase synchronization values.
        """

        # arguments of the modes
        arg_is = np.angle(self.Modes[:, int(pos_i / 2)])
        arg_js = np.angle(self.Modes[:, int(pos_j / 2)])

        # frequently used variables
        cos_is = np.cos(arg_is)
        cos_js = np.cos(arg_js)
        sin_is = np.sin(arg_is)
        sin_js = np.sin(arg_js)

        # transformation for ith mode momentum quadrature
        p_i_prime_2s = np.multiply(sin_is**2, self.Corrs[:, pos_i, pos_i]) \
                        - np.multiply(np.multiply(sin_is, cos_is), self.Corrs[:, pos_i, pos_i + 1]) \
                        - np.multiply(np.multiply(cos_is, sin_is), self.Corrs[:, pos_i + 1, pos_i]) \
                        + np.multiply(cos_is**2, self.Corrs[:, pos_i + 1, pos_i + 1])

        # transformation for jth mode momentum quadrature
        p_j_prime_2s = np.multiply(sin_js**2, self.Corrs[:, pos_j, pos_j]) \
                        - np.multiply(np.multiply(sin_js, cos_js), self.Corrs[:, pos_j, pos_j + 1]) \
                        - np.multiply(np.multiply(cos_js, sin_js), self.Corrs[:, pos_j + 1, pos_j]) \
                        + np.multiply(cos_js**2, self.Corrs[:, pos_j + 1, pos_j + 1])

        # transformation for intermode momentum quadratures
        p_i_p_j_primes = np.multiply(np.multiply(sin_is, sin_js), self.Corrs[:, pos_i, pos_j]) \
                        - np.multiply(np.multiply(sin_is, cos_js), self.Corrs[:, pos_i, pos_j + 1]) \
                        - np.multiply(np.multiply(cos_is, sin_js), self.Corrs[:, pos_i + 1, pos_j]) \
                        + np.multiply(np.multiply(cos_is, cos_js), self.Corrs[:, pos_i + 1, pos_j + 1])

        # square difference between momentum quadratures
        p_minus_prime_2s = 0.5 * (p_i_prime_2s + p_j_prime_2s - 2 * p_i_p_j_primes)

        # quantum phase synchronization values
        return 0.5 / p_minus_prime_2s

def get_average_amplitude_difference(Modes):
    """Method to obtain the average amplitude differences for two specific modes [7]_.
    
    Parameters
    ----------
    Modes : numpy.ndarray
        The two specific modes with shape ``(dim, 2)``.

    Returns
    -------
    diff_a : float
        The average amplitude difference.
    """

    # validate modes
    assert Modes is not None and isinstance(Modes, (list, np.ndarray)) and np.shape(Modes)[1] == 2, "Parameter ``Modes`` should be a list or NumPy array with dimension ``(dim, 2)``"

    # get means
    means = np.mean(Modes, axis=0)

    # average amplitude difference
    return np.mean([np.linalg.norm(modes[0] - means[0]) - np.linalg.norm(modes[1]- means[1]) for modes in Modes])

def get_average_phase_difference(Modes):
    """Method to obtain the average phase differences for two specific modes [4]_.
    
    Parameters
    ----------
    Modes : numpy.ndarray
        The two specific modes with shape ``(dim, 2)``.

    Returns
    -------
    diff_p : float
        The average phase difference.
    """

    # validate modes
    assert Modes is not None and isinstance(Modes, (list, np.ndarray)) and np.shape(Modes)[1] == 2, "Parameter ``Modes`` should be a list or NumPy array with dimension ``(dim, 2)``"

    # get means
    means = np.mean(Modes, axis=0)

    # average phase difference
    return np.mean([np.angle(modes[0] - means[0]) - np.angle(modes[1]- means[1]) for modes in Modes])

def get_bifurcation_amplitudes(Modes):
    """Method to obtain the bifurcation amplitudes of the modes.
    
    Parameters
    ----------
    Modes : numpy.ndarray
        The mode amplitudes in the trajectory with shape ``(dim, num_modes)``.

    Returns
    -------
    Amps : list
        The bifurcation amplitudes of the modes. The first ``num_modes`` arrays contain the bifurcation amplitudes of the real parts of the modes; the next ``num_modes`` arrays contain those of the imaginary parts.
    """

    # validate modes
    assert Modes is not None and isinstance(Modes, (list, np.ndarray)) and len(np.shape(Modes)) == 2, "Parameter ``Modes`` should be a list or NumPy array with dimension ``(dim, num_modes)``"

    # convert to real
    Modes_real = np.concatenate((np.real(Modes), np.imag(Modes)), axis=1)

    # calculate gradients
    grads = np.gradient(Modes_real, axis=0)

    # get indices where the derivative changes sign
    idxs = grads[:-1, :] * grads[1:, :] < 0

    Amps = []
    for i in range(idxs.shape[1]):
        # collect all crests and troughs
        extremas = Modes_real[:-1, i][idxs[:, i]]
        # save absolute values of differences
        Amps.append(np.abs(extremas[:-1] - extremas[1:]))

    return Amps

def get_correlation_Pearson(Modes):
    r"""Method to obtain the Pearson correlation coefficient for two specific modes [6]_.

    .. math::

        C_{ij} = \frac{\Sigma_{t} \langle \mathcal{O}_{i} (t) \mathcal{O}_{j} (t) \rangle}{\sqrt{\Sigma_{t} \langle \mathcal{O}_{i}^{2} (t) \rangle} \sqrt{\Sigma_{t} \langle \mathcal{O}_{j}^{2} (t) \rangle}}

    where :math:`\mathcal{O}_{i}` and :math:`\mathcal{O}_{j}` are the corresponding modes.
    
    Parameters
    ----------
    Modes : numpy.ndarray
        The two specific modes with shape ``(dim, 2)``.

    Returns
    -------
    corr_P : float
        Pearson correlation coefficient.
    """

    # validate modes
    assert Modes is not None and isinstance(Modes, (list, np.ndarray)) and np.shape(Modes)[1] == 2, "Parameter ``Modes`` should be a list or NumPy array with dimension ``(dim, 2)``"

    # get means
    means = np.mean(Modes, axis=0)
    mean_ii = np.mean([np.linalg.norm(modes[0] - means[0])**2 for modes in Modes])
    mean_ij = np.mean([np.linalg.norm(modes[0] - means[0]) * np.linalg.norm(modes[1] - means[1]) for modes in Modes])
    mean_jj = np.mean([np.linalg.norm(modes[1] - means[1])**2 for modes in Modes])

    # Pearson correlation coefficient
    return mean_ij / np.sqrt(mean_ii * mean_jj)

def get_Wigner_distributions_single_mode(Corrs, params, cb_update=None):
    """Method to obtain single-mode Wigner distribitions.
    
    Parameters
    ----------
    Corrs : numpy.ndarray
        Quadrature quadrature correlations with shape ``(dim, 2 * num_modes, 2 * num_modes)``.
    params : dict
        Parameters of the solver. Available options are:

        ================    ====================================================
        key                 value
        ================    ====================================================
        'show_progress'     (*bool*) option to display the progress of the solver. Default is ``False``.
        'indices'           (*list* or *tuple*) indices of the modes as a list. Default is ``[0]``.
        'wigner_xs'         (*list*) X-axis values.
        'wigner_ys'         (*list*) Y-axis values.
        ================    ====================================================
    cb_update : callable, optional
        Callback function to update status and progress, formatted as ``cb_update(status, progress, reset)``, where ``status`` is a string, ``progress`` is a float and ``reset`` is a boolean.
    
    Returns
    -------
    Wigners : numpy.ndarray
        Single-mode Wigner distributions of shape ``(dim_1, dim_0, p_dim, q_dim)``, where ``dim_1`` and ``dim_0`` are the first dimensions of the correlations and the indices respectively.
    """

    # validate correlations
    validate_Modes_Corrs(
        Corrs=Corrs,
        is_corrs_required=True
    )

    # validate indices
    indices = params.get('indices', [0])
    assert isinstance(indices, Union[list, tuple].__args__), "Solver parameter ``'indices'`` should be a ``list`` or ``tuple`` of mode indices"

    # validate axes
    xs = params.get('wigner_xs', None)
    ys = params.get('wigner_ys', None)
    for val in [xs, ys]:
        assert val is not None and isinstance(val, (list, np.ndarray)), "Solver parameters ``'wigner_xs'`` and ``'wigner_ys'`` should be either NumPy arrays or ``list``"
    # handle list
    xs = np.array(xs, dtype=np.float_) if isinstance(xs, list) else xs
    ys = np.array(ys, dtype=np.float_) if isinstance(xs, list) else ys

    # extract frequently used variables
    show_progress = params.get('show_progress', False)
    dim_m = len(indices)
    dim_c = len(Corrs)
    dim_w = len(ys) * len(xs)
    # get column vectors and row vectors
    _X, _Y = np.meshgrid(xs, ys)
    _dim = (ys.shape[0], xs.shape[0], 1, 1)
    Vects = np.concatenate((np.reshape(_X, _dim), np.reshape(_Y, _dim)), axis=2)
    Vects_t = np.transpose(Vects, axes=(0, 1, 3, 2))

    # initialize measures
    Wigners = np.zeros((dim_c, dim_m, ys.shape[0], xs.shape[0]), dtype=np.float_)

    # iterate over indices
    for j in range(dim_m):
        # get position
        pos = 2 * indices[j]

        # reduced correlation matrices
        V_pos = Corrs[:, pos:pos + 2, pos:pos + 2]
        invs = np.linalg.pinv(V_pos)
        dets = np.linalg.det(V_pos)

        # calculate dot product of inverse and column vectors
        _dots = np.transpose(np.dot(invs, Vects), axes=(4, 2, 3, 1, 0))[0]

        # get Wigner distributions
        for idx_y in range(len(ys)):
            for idx_x in range(len(xs)):
                # display progress
                if show_progress and cb_update is not None:
                    _index_status = str(j + 1) + "/" + str(dim_m)
                    cb_update(
                        status="-" * (18 - len(_index_status)) + "Obtaining Wigners (" + _index_status + ")",
                        progress=(idx_y * len(xs) + idx_x) / dim_w,
                        reset=False
                    )
                # wigner function
                Wigners[:, j, idx_y, idx_x] = np.exp(- 0.5 * np.dot(Vects_t[idx_y, idx_x], _dots[idx_y, idx_x])[0]) / 2.0 / np.pi / np.sqrt(dets)

    return Wigners

def get_Wigner_distributions_two_mode(Corrs, params, cb_update=None):
    """Method to obtain two-mode Wigner distribitions.
    
    Parameters
    ----------
    Corrs : numpy.ndarray
        Quadrature quadrature correlations with shape ``(dim, 2 * num_modes, 2 * num_modes)``.
    params : dict
        Parameters of the solver. Available options are:

        ================    ====================================================
        key                 value
        ================    ====================================================
        'show_progress'     (*bool*) option to display the progress of the solver. Default is ``False``.
        'indices'           (*list* or *tuple*) list of indices of the modes and their quadratures as tuples or lists. Default is ``[(0, 0), (1, 0)]``.
        'wigner_xs'         (*list*) X-axis values.
        'wigner_ys'         (*list*) Y-axis values.
        ================    ====================================================
    cb_update : callable, optional
        Callback function to update status and progress, formatted as ``cb_update(status, progress, reset)``, where ``status`` is a string, ``progress`` is a float and ``reset`` is a boolean.
    
    Returns
    -------
    Wigners : numpy.ndarray
        Two-mode Wigner distributions of shape ``(dim_0, p_dim, q_dim)``, where ``dim_0`` is the first dimension of the correlations.
    """

    # validate correlations
    validate_Modes_Corrs(
        Corrs=Corrs,
        is_corrs_required=True
    )

    # validate indices
    indices = params.get('indices', [(0, 0), (1, 0)])
    assert isinstance(indices, Union[list, tuple].__args__), "Solver parameter ``'indices'`` should be a ``list`` or ``tuple`` with each element being a ``list`` or ``tuple`` of the mode index and its quadrature index"
    for idxs in indices:
        assert isinstance(idxs, Union[list, tuple].__args__), "Each element of the indices should be a ``list`` or ``tuple`` of the mode index and its quadrature index"

    # validate axes
    xs = params.get('wigner_xs', None)
    ys = params.get('wigner_ys', None)
    for val in [xs, ys]:
        assert val is not None and isinstance(val, (list, np.ndarray)), "Solver parameters ``'wigner_xs'`` and ``'wigner_ys'`` should be either NumPy arrays or ``list``"
    # handle list
    xs = np.array(xs, dtype=np.float_) if isinstance(xs, list) else xs
    ys = np.array(ys, dtype=np.float_) if isinstance(xs, list) else ys

    # extract frequently used variables
    show_progress = params.get('show_progress', False)
    indices = params.get('indices', [0])
    dim_c = len(Corrs)
    dim_w = len(ys) * len(xs)
    pos_i = 2 * indices[0][0]
    pos_j = 2 * indices[1][0]
    # get column vectors and row vectors
    _X, _Y = np.meshgrid(xs, ys)
    _dim = (ys.shape[0], xs.shape[0], 1, 1)
    Vects_a = np.concatenate((np.reshape(_X, _dim), np.zeros(_dim, dtype=np.float_)), axis=2) if indices[0][1] == 0 else np.concatenate((np.zeros(_dim, dtype=np.float_), np.reshape(_X, _dim)), axis=2)
    Vects_b = np.concatenate((np.reshape(_Y, _dim), np.zeros(_dim, dtype=np.float_)), axis=2) if indices[1][1] == 0 else np.concatenate((np.zeros(_dim, dtype=np.float_), np.reshape(_Y, _dim)), axis=2)
    Vects = np.concatenate((Vects_a, Vects_b), axis=2)
    Vects_t = np.transpose(Vects, axes=(0, 1, 3, 2))

    # initialize measures
    Wigners = np.zeros((dim_c, ys.shape[0], xs.shape[0]), dtype=np.float_)

    # correlation matrix of the ith mode
    As = Corrs[:, pos_i:pos_i + 2, pos_i:pos_i + 2]
    # correlation matrix of the jth mode
    Bs = Corrs[:, pos_j:pos_j + 2, pos_j:pos_j + 2]
    # correlation matrix of the intermodes
    Cs = Corrs[:, pos_i:pos_i + 2, pos_j:pos_j + 2]

    # get transposes matrices
    C_Ts = np.array(np.transpose(Cs, axes=(0, 2, 1)))

    # reduced correlation matrices
    V_pos = np.concatenate((np.concatenate((As, Cs), axis=2), np.concatenate((C_Ts, Bs), axis=2)), axis=1)
    invs = np.linalg.pinv(V_pos)
    dets = np.linalg.det(V_pos)

    # calculate dot product of inverse and column vectors
    _dots = np.transpose(np.dot(invs, Vects), axes=(4, 2, 3, 1, 0))[0]

    # get Wigner distributions
    for idx_y in range(len(ys)):
        for idx_x in range(len(xs)):
            # display progress
            if show_progress and cb_update is not None:
                cb_update(
                    status="-" * 21 + "Obtaining Wigners",
                    progress=(idx_y * len(xs) + idx_x) / dim_w,
                    reset=False
                )
            # wigner function
            Wigners[:, idx_y, idx_x] = np.exp(- 0.5 * np.dot(Vects_t[idx_y, idx_x], _dots[idx_y, idx_x])[0]) / 4.0 / np.pi**2 / np.sqrt(dets)

    return Wigners

def validate_Modes_Corrs(Modes=None, Corrs=None, is_modes_required:bool=False, is_corrs_required:bool=False):
    """Function to validate the modes and correlations.

    At least one of ``Modes`` or ``Corrs`` should be non-``None``.
    
    Parameters
    ----------
    Modes : list or numpy.ndarray, optional
        Classical modes with shape ``(dim, num_modes)``.
    Corrs : list or numpy.ndarray, optional
        Quadrature quadrature correlations with shape ``(dim, 2 * num_modes, 2 * num_modes)``.
    is_modes_required : bool, optional
        Option to set ``Modes`` as required.
    is_corrs_required : bool, optional
        Option to set ``Corrs`` as required.

    Returns
    -------
    Modes : numpy.ndarray
        Classical modes with shape ``(dim, num_modes)``.
    Corrs : numpy.ndarray
        Quadrature quadrature correlations with shape ``(dim, 2 * num_modes, 2 * num_modes)``.
    """

    # handle null
    assert Modes is not None or Corrs is not None, "At least one of ``Modes`` or ``Corrs`` should be non-``None``"

    # check requirements
    assert Modes is not None if is_modes_required else True, "Missing required parameter ``Modes``"
    assert Corrs is not None if is_corrs_required else True, "Missing required parameter ``Corrs``"

    # handle list
    Modes  = np.array(Modes, dtype=np.complex_) if Modes is not None and isinstance(Modes, list) else Modes
    Corrs  = np.array(Corrs, dtype=np.float_) if Corrs is not None and isinstance(Corrs, list) else Corrs

    # validate shapes
    assert len(Modes.shape) == 2 if Modes is not None else True, "``Modes`` should be of shape ``(dim, num_modes)``"
    assert (len(Corrs.shape) == 3 and Corrs.shape[1] == Corrs.shape[2]) if Corrs is not None else True, "``Corrs`` should be of shape ``(dim, 2 * num_modes, 2 * num_modes)``"
    assert 2 * Modes.shape[1] == Corrs.shape[1] if Modes is not None and Corrs is not None else True, "Shape mismatch for ``Modes`` and ``Corrs``; expected shapes are ``(dim, num_modes)`` and ``(dim, 2 * num_modes, 2 * num_modes)``"

    return Modes, Corrs

def validate_As_Coeffs(As=None, Coeffs=None):
    """Function to validate the drift matrices and the coefficients.
    
    Parameters
    ----------
    As : list or numpy.ndarray, optional
        Drift matrix.
    Coeffs : list or numpy.ndarray, optional
        Coefficients of the characteristic equation.

    Returns
    -------
    As : numpy.ndarray
        Drift matrix.
    Coeffs : numpy.ndarray
        Coefficients of the characteristic equation.
    """

    # check non-empty
    assert As is not None or Coeffs is not None, "At least one of ``As`` and ``Coeffs`` should be non-``None``"

    # if drift matrices are given
    if As is not None:
        # validate drift matrix
        assert isinstance(As, Union[list, np.ndarray].__args__), "``As`` should be of type ``list`` or ``numpy.ndarray``"
        # convert to numpy array
        As = np.array(As, dtype=np.float_) if isinstance(As, list) else As
        # validate shape
        assert len(As.shape) == 3 and As.shape[1] == As.shape[2], "``As`` should be of shape ``(dim_0, 2 * num_modes, 2 * num_modes)``"
    # if coefficients are given
    else:
        # validate coefficients
        assert isinstance(Coeffs, Union[list, np.ndarray].__args__), "``Coeffs`` should be of type ``list`` or ``numpy.ndarray``"
        # convert to numpy array
        Coeffs = np.array(Coeffs, dtype=np.float_) if isinstance(Coeffs, list) else Coeffs
        # validate shape
        assert len(Coeffs.shape) == 2, "``Coeffs`` should be of shape ``(dim_0, 2 * num_modes + 1)``"

    return As, Coeffs

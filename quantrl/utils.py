#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for input output operations."""

__name__    = 'quantrl.utils'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-06-02"
__updated__ = "2024-07-06"

# dependencies
from glob import escape
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gc
import numpy as np
import os

class SaveOnBestMeanRewardCallback(BaseCallback):
    """Utility callback class to save the best mean reward.

    Parameters
    ----------
    update_steps: int
        Number of action steps after which the mean is updated.
    log_dir: str
        Directory where the data is stored.
    action_steps: int
        Number of action steps in each episode.
    """

    def __init__(self, update_steps: int, log_dir: str, action_steps: int):
        """Class constructor for SaveOnBestMeanRewardCallback."""

        # initialize BaseCallback class
        super().__init__(verbose=1)

        # set attributes
        self.update_steps = update_steps
        self.log_dir = log_dir
        self.action_steps = action_steps
        self.best_traj = -1
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Method on intialization."""

        # create folders
        if self.log_dir is not None:
            os.makedirs(self.log_dir + 'models/', exist_ok=True)
            os.makedirs(self.log_dir + 'buffers/', exist_ok=True)

    def _on_step(self) -> bool:
        """Method on call."""

        if self.n_calls % self.update_steps == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(escape(self.log_dir)), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Best mean reward: {self.best_mean_reward:.6f} at #{self.best_traj:6d}\nCurr mean reward: {mean_reward:.6f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_traj = int(self.n_calls / self.action_steps)
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model and replay buffer...")
                    self.model.save(self.log_dir + 'models/best.zip')
                    self.model.save_replay_buffer(self.log_dir + 'buffers/best.zip')
            gc.collect()
        return True
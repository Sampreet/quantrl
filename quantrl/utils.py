#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for input output operations."""

__name__    = 'quantrl.utils'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-06-02"
__updated__ = "2024-08-09"

# dependencies
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy
import gc
import numpy as np
import os
import pandas
import time

class SaveOnBestMeanRewardCallback(BaseCallback):
    """Utility callback class to save the best mean reward.

    Parameters
    ----------
    update_steps: int
        Number of action steps after which the mean is updated.
    log_dir: str
        Directory where the data is stored.
    n_episodes: int
        Total number of episodes.
    episode_start: int
        Starting index of the episodes.
    steps_per_episode: int
        Number of action steps in each episode.
    average_over: int, default=100
        Number of episodes over which the average reward is calculated.
    """

    def __init__(self,
        update_steps:int,
        log_dir:str,
        n_episodes:int,
        episode_start:int,
        steps_per_episode:int,
        average_over:int=100
    ):
        """Class constructor for SaveOnBestMeanRewardCallback."""

        # initialize BaseCallback class
        super().__init__(verbose=1)

        # set attributes
        self.update_steps = update_steps
        self.steps_per_episode = steps_per_episode
        self.log_dir = log_dir
        self.episode_start = episode_start
        self.n_episodes = n_episodes
        self.episode_best = -1
        self.reward_best = -np.inf
        self.average_over = average_over
        self.t_start = time.time()

        # initialize file to save mean rewards
        with open(self.log_dir + f'reward_mean_{self.episode_start}_{self.n_episodes - 1}.txt', 'w') as file:
            s = f'{"time":>14} {"n_steps":>12} {"episode_curr":>12} {"reward_curr":>14} {"episode_best":>12} {"reward_best":>14}\n'
            file.write(s)
            file.close()

    def _init_callback(self) -> None:
        """Method on intialization."""

        # create folders
        if self.log_dir is not None:
            os.makedirs(self.log_dir + 'models/', exist_ok=True)
            os.makedirs(self.log_dir + 'buffers/', exist_ok=True)

    def _on_step(self) -> bool:
        """Method on call."""

        if self.n_calls % self.update_steps == 0:
            # retrieve reward data from monitor file
            with open(self.log_dir + f'learning_{self.episode_start}_{self.n_episodes - 1}.monitor.csv') as file_handler:
                file_handler.readline()
                xs, ys = ts2xy(pandas.read_csv(file_handler, index_col=None), 'timesteps')

            if len(xs) > 0:
                # current mean reward
                reward_curr = np.mean(ys[-self.average_over:])
                # current episode
                episode_curr = int(self.n_calls / self.steps_per_episode)

                # update console
                if self.verbose >= 1:
                    print(f"Best mean reward: {self.reward_best:.6f} at #{self.episode_best:6d}\nCurr mean reward: {reward_curr:.6f}")

                # New best model, you could save the agent here
                if reward_curr > self.reward_best:
                    self.episode_best = episode_curr
                    self.reward_best = reward_curr

                    # update console
                    if self.verbose >= 1:
                        print(f"Saving new best model and replay buffer...")

                    # save model and replay buffer
                    self.model.save(self.log_dir + f'models/best_{self.n_episodes - 1}.zip')
                    self.model.save_replay_buffer(self.log_dir + f'buffers/best_{self.n_episodes - 1}.zip')

                # save reward data
                with open(self.log_dir + f'reward_mean_{self.episode_start}_{self.n_episodes - 1}.txt', 'a') as file:
                    file.write(f'{time.time() - self.t_start:14.03f} {self.n_calls:12d} {episode_curr:12d} {reward_curr:14.06f} {self.episode_best:12d} {self.reward_best:14.06f}\n')
                    file.close()
            gc.collect()
        return True
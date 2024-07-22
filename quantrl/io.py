#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for input output operations."""

__name__    = 'quantrl.io'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-12-07"
__updated__ = "2024-07-22"

# dependencies
from threading import Thread
from tqdm.rich import tqdm
import gc
import numpy as np
import os

# TODO: Implement ConsoleIO
# TODO: Add ``load_parts`` method

class FileIO(object):
    """Handler for file input-output.

    Initializes ``cache`` to ``None`` and ``index`` to ``-1``.
    Subsequent calls to ``update_cache`` allocates ``cache`` and updates ``index``.
    The parent needs to implement the ``close`` method to cache the final file.

    Parameters
    ----------
    disk_cache_dir: str
        Directory path for the disk cache. If the value of ``disk_cache_size`` is ``0``, the then this parameter serves as the file path for a single disk cache, else the cache is dumped in parts.
    cache_dump_interval: int, default=100
        Number of steps to update the cache before dumping it to disk. Should be a positive integer.
    """

    def __init__(self,
        disk_cache_dir:str,
        cache_dump_interval:int=100
    ):
        """Class constructor for FileIO."""

        # set attributes
        assert type(cache_dump_interval) is int and cache_dump_interval > 0, "parameter ``disk_cache_size`` should be a positive integer"
        self.disk_cache_dir = disk_cache_dir
        self.cache_dump_interval = cache_dump_interval
        try:
            os.makedirs(self.disk_cache_dir, exist_ok=True)
        except OSError:
            pass

        # initialize variables
        self.cache = None
        self.index = -1

    def dump_part_async(self,
        data:np.ndarray,
        batch_idx:int,
        part_idx:int
    ):
        """Method to dump a batch of data to disk asynchronously.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            Data to dump.
        batch_idx: int
            Index of the batch.
        part_idx: int
            Index of the part.
        """

        # save as compressed NumPy data from another thread
        thread = Thread(target=np.savez_compressed, args=(self.disk_cache_dir  + '/' + '_'.join([
            str(batch_idx * self.cache_dump_interval),
            str((batch_idx + 1) * self.cache_dump_interval - 1),
            str(part_idx)
        ]) + '.npz', data))
        thread.start()
        thread.join()

        # clear cache
        del data
        gc.collect()

    def update_cache(self,
        data:np.ndarray
    ):
        """Method to update the cache with data.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            Data to cache.
        """

        # update list
        if self.cache is None:
            self.cache = np.zeros((self.cache_dump_interval, *data.shape), dtype=np.float_)

        self.index += 1
        self.cache[self.index % self.cache_dump_interval] = data

        # dump cache
        if self.index != 0 and (self.index + 1) % self.cache_dump_interval == 0:
            self._dump_cache_async(
                idx_start=self.index - self.cache_dump_interval + 1,
                idx_end=self.index
            )

    def _dump_cache_async(self,
        idx_start:int,
        idx_end:int
    ):
        """Method to dump cache to disk asynchronously.

        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        """

        # save as compressed NumPy data from another thread
        thread = Thread(target=np.savez_compressed, args=(self.disk_cache_dir + '/' + str(idx_start) + '_' + str(idx_end) + '.npz', self.cache))
        thread.start()

        # clear cache
        del self.cache
        self.cache = None
        gc.collect()

    def get_disk_cache(self,
        idx_start:int=0,
        idx_end:int=-1,
        idxs:list=None
    ):
        """Method to return select disk-cached data between a given set of indices.

        Parameters
        ----------
        idx_start: int, default=0
            Starting index for the part file.
        idx_end: int, default=-1
            Ending index for the part file.
        idxs: list or slice, default=None
            Indices of the data values required. If ``None``, all data is returned.
        """

        # iterate over parts
        self.cache_list = list()
        for i in tqdm(
            range(int(idx_start / self.cache_dump_interval) * self.cache_dump_interval, idx_end + 1, self.cache_dump_interval),
            desc="Loading",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            # update end index
            _idx_e = i + self.cache_dump_interval - 1
            # update cache list
            _cache = self._load_cache(
                idx_start=i,
                idx_end=_idx_e
            )
            self.cache_list += [_cache[:, :, idxs].copy() if idxs is not None else _cache.copy()]
            # clear loaded cache
            del _cache
            gc.collect()

        return np.concatenate(self.cache_list)[idx_start % self.cache_dump_interval:]

    def _load_cache(self,
        idx_start:int,
        idx_end:int
    ):
        """Method to load cache from disk.

        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        """

        # load part or single cache file
        return np.load(self.disk_cache_dir + '/' + str(idx_start) + '_' + (str(idx_end) if idx_end != -1 else '*') + '.npz')['arr_0']

    def save_data(self,
        data:np.ndarray,
        file_name:str
    ):
        """Method to save data to a file.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            Data to save.
        file_name: str
            Name of the file.
        """

        np.savez_compressed(file_name + '.npz', data)

    def load_data(self,
        file_name:str
    ):
        """Method to load data from a file.

        Parameters
        ----------
        file_name: str
            Name of the file.

        Returns
        -------
        data: :class:`numpy.ndarray`
            Data loaded from the file. Returns `None` if the file does not exist.
        """

        if os.path.isfile(file_name + '.npz'):
            return np.load(file_name + '.npz')['arr_0']
        return None

    def close(self,
        dump_cache=True
    ):
        """Method to close FileIO.

        Parameters
        ----------
        dump_cache: bool, default=True
            Option to dump the cache.
        """

        if dump_cache and self.cache is not None:
            _idx_s = self.index - (self.index + 1) % self.cache_dump_interval + 1
            self._dump_cache_async(
                idx_start=_idx_s,
                idx_end=_idx_s + self.cache_dump_interval - 1
            )

        # clean
        del self
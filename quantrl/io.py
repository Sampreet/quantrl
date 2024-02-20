#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for input output operations."""

__name__    = 'quantrl.io'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-12-07"
__updated__ = "2024-02-17"

# dependencies
from tqdm import tqdm
import numpy as np
import os

# TODO: Implement ConsoleIO

class FileIO(object):
    """Handler for file input-output.

    Initializes ``cache_in_parts``, ``cache`` and ``index``.
    The parent needs to implement the ``close`` method.

    Parameters
    ----------
    disk_cache_dir: str
        Directory path for the disk cache. If the value of ``max_cache_size`` is ``0``, the then this parameter serves as the file path for a single disk cache, else the cache is dumped in parts.
    max_cache_size: int, default=100
        Maximum number of data points to cache in memory before dumping to disk. If ``0``, the cache is dumped in parts.
    """

    def __init__(self,
        disk_cache_dir:str,
        max_cache_size:int=100
    ):
        """Class constructor for FileIO."""

        # constants
        self.disk_cache_dir = disk_cache_dir
        self.max_cache_size = max_cache_size
        self.cache_in_parts = False if self.max_cache_size == 0 else True
        try:
            os.makedirs(self.disk_cache_dir if self.cache_in_parts else self.disk_cache_dir[:self.disk_cache_dir.rfind('/')], exist_ok=True)
        except OSError:
            pass

        # initialize variables
        self.cache = list()
        self.index = -1

    def update_cache(self,
        data
    ):
        """Method to update the cache with data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to dump.
        """

        # update list
        self.cache.append(data)
        self.index += 1

        # dump cache
        if self.cache_in_parts and self.index != 0 and (self.index + 1) % self.max_cache_size == 0:
            self._dump_cache(
                idx_start=self.index - self.max_cache_size + 1,
                idx_end=self.index
            )

    def _dump_cache(self,
        idx_start:int,
        idx_end:int
    ):
        """Method to dump cache to disk.

        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        """

        # save as compressed NumPy data
        np.savez_compressed((self.disk_cache_dir + '/' + str(idx_start) + '_' + str(idx_end) + '.npz') if self.cache_in_parts else (self.disk_cache_dir + '.npz'), np.array(self.cache))

        # reset cache
        del self.cache
        self.cache = list()

    def close(self):
        """Method to close FileIO."""

        # if already done
        if len(self.cache) == 0:
            return
        # if single cache file
        elif not self.cache_in_parts:
            self._dump_cache(
                idx_start=None,
                idx_end=None
            )
        # else cache final part
        else:
            _idx_s = self.index - (self.index + 1) % self.max_cache_size + 1
            self._dump_cache(
                idx_start=_idx_s,
                idx_end=_idx_s + self.max_cache_size - 1
            )

        # clean
        del self

    def get_disk_cache(self,
        idx_start:int=0,
        idx_end:int=-1,
        idxs:list=None
    ):
        """Method to return select disk-cached data between a given set of indices.

        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        idxs: list
            Indices of the data values required. If ``None``, all data is returned.
        """

        # single cache file
        if not self.cache_in_parts:
            return self._load_cache(
                idx_start=0,
                idx_end=-1
            )[idx_start:idx_end]

        # if cached in parts
        self.cache_list = list()
        for i in tqdm(
            range(int(idx_start / self.max_cache_size) * self.max_cache_size, idx_end + 1, self.max_cache_size),
            desc="Loading",
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            # update end index
            _idx_e = i + self.max_cache_size - 1
            # update cache list
            _cache = self._load_cache(
                idx_start=i,
                idx_end=_idx_e
            )
            self.cache_list += [_cache[:, :, idxs] if idxs is not None else _cache]

        return np.concatenate(self.cache_list)[idx_start % self.max_cache_size:]

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
        return np.load((self.disk_cache_dir + '/' + str(idx_start) + '_' + (str(idx_end) if idx_end != -1 else '*') + '.npz') if self.cache_in_parts else (self.disk_cache_dir + '.npz'))['arr_0']
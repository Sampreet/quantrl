#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module for input output operations."""

__name__    = 'quantrl.io'
__authors__ = ['Sampreet Kalita']
__created__ = '2023-12-07'
__updated__ = '2024-01-07'

# dependencies
from tqdm import tqdm
import numpy as np
import os

class BaseIO(object):
    """Base class for file I/O.
    
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
        """Class constructor for BaseIO."""

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

    def close(self):
        """Method to clean up."""

        # if already done
        if len(self.cache) == 0:
            return
        elif not self.cache_in_parts:
            self._dump_cache(
                idx_start=None,
                idx_end=None
            )
        else:
            self._dump_cache(
                idx_start=self.index - (self.index + 1) % self.max_cache_size + 1,
                idx_end=self.index
            )
        del self

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

        np.savez_compressed((self.disk_cache_dir + '/' + str(idx_start) + '_' + str(idx_end) + '.npz') if self.cache_in_parts else (self.disk_cache_dir + '.npz'), np.array(self.cache))
        del self.cache
        self.cache = list()

    def get_disk_cache(self,
        idx_start:int=0,
        idx_end:int=-1
    ):
        """Method to return all disk cached data between a given set of indices.
        
        Parameters
        ----------
        idx_start: int
            Starting index for the part file.
        idx_end: int
            Ending index for the part file.
        """

        # single cache file
        if not self.cache_in_parts:
            return self._load_cache(
                idx_start=idx_start,
                idx_end=idx_end
            )[idx_start:idx_end]
        # if cached in parts
        self.cache_list = list()
        for i in tqdm(
            range(int(idx_start / self.max_cache_size) * self.max_cache_size, idx_end + 1, self.max_cache_size),
            desc='Loading',
            leave=False,
            mininterval=0.5,
            disable=False
        ):
            _idx_e = i + self.max_cache_size - 1
            # if not divisible
            if _idx_e > idx_end:
                _idx_e = idx_end
            self.cache_list += [self._load_cache(
                idx_start=i,
                idx_end=_idx_e
            )]
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

        return np.load((self.disk_cache_dir + '/' + str(idx_start) + '_' + str(idx_end) + '.npz') if self.cache_in_parts else (self.disk_cache_dir + '.npz'))['arr_0']
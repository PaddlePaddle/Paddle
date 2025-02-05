#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import queue
import sys
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle

from ...framework import core
from ..multiprocess_utils import (
    MP_STATUS_CHECK_INTERVAL,
    CleanupFuncRegistrar,
    _cleanup_mmap,
)
from .fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from .flat import _flatten_batch

if TYPE_CHECKING:
    from paddle.io import Dataset


class _IterableDatasetStopIteration:
    def __init__(self, worker_id):
        self.worker_id = worker_id


class _ResumeIteration:
    pass


class _DatasetKind:
    MAP = 0
    ITER = 1

    @staticmethod
    def create_fetcher(
        kind, dataset, auto_collate_batch, collate_fn, drop_last
    ):
        if kind == _DatasetKind.MAP:
            return _MapDatasetFetcher(
                dataset, auto_collate_batch, collate_fn, drop_last
            )
        elif kind == _DatasetKind.ITER:
            return _IterableDatasetFetcher(
                dataset, auto_collate_batch, collate_fn, drop_last
            )
        else:
            raise NotImplementedError(f"unknown Dataset kind {kind}")


class ParentWatchDog:
    def __init__(self):
        self._parent_pid = os.getppid()
        self._parent_alive = True

    def is_alive(self):
        if self._parent_alive:
            self._parent_alive = os.getppid() == self._parent_pid
        return self._parent_alive


# worker information for each workers, used for splitting data copy
# for IteratorDataset in worker processes.
_worker_info = None


def get_worker_info() -> WorkerInfo:
    """
    Get DataLoader worker process information function, this function is
    used to split data copy in worker process for IterableDataset
    (see :code:`paddle.io.IterableDataset`), worker information contains
    following fields:

    :attr:`num_workers`: total worker process number, see `paddle.io.DataLoader`

    :attr:`id`: the worker process id, count from 0 to :attr:`num_workers - 1`

    :attr:`dataset`: the dataset object in this worker process

    Returns:
        WorkerInfo: an instance of WorkerInfo which contains fields above.

    Notes:
        For more usage and examples, please see :code:`paddle.io.IterableDataset`

    Example:

        .. code-block:: python

            >>> import math
            >>> import paddle
            >>> import numpy as np
            >>> from paddle.io import IterableDataset, DataLoader, get_worker_info

            >>> class SplitedIterableDataset(IterableDataset): # type: ignore[type-arg]
            ...     def __init__(self, start, end):
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         worker_info = get_worker_info()
            ...         if worker_info is None:
            ...             iter_start = self.start
            ...             iter_end = self.end
            ...         else:
            ...             per_worker = int(
            ...                 math.ceil((self.end - self.start) / float(
            ...                     worker_info.num_workers)))
            ...             worker_id = worker_info.id
            ...             iter_start = self.start + worker_id * per_worker
            ...             iter_end = min(iter_start + per_worker, self.end)
            ...
            ...         for i in range(iter_start, iter_end):
            ...             yield np.array([i])
            ...
            >>> place = paddle.CPUPlace()
            >>> dataset = SplitedIterableDataset(start=2, end=9)
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     places=place,
            ...     num_workers=2,
            ...     batch_size=1,
            ...     drop_last=True)
            ...
            >>> for data in dataloader:
            ...     print(data) # doctest: +SKIP("The output depends on the environment.")
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[6]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[7]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[4]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[8]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[5]])

    """
    return _worker_info


class WorkerInfo:
    num_workers: int
    id: int
    dataset: Dataset[Any]
    seed: int

    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(
                f"Cannot assign attributes to {self.__class__.__name__} objects"
            )
        return super().__setattr__(key, val)


class _WorkerException:
    def __init__(self, worker_id, exc_info=None):
        self.worker_id = worker_id
        exc_info = exc_info or sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def reraise(self):
        msg = f"DataLoader worker({self.worker_id}) caught {self.exc_type.__name__} with message:\n{self.exc_msg}"
        if getattr(self.exc_type, "message", None):
            raise self.exc_type(message=msg)
        raise self.exc_type(msg)


# The function `_generate_states` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# Here is the copyright:

# SeedSequence is derived from Melissa E. O'Neill's C++11 `std::seed_seq`
# implementation, as it has a lot of nice properties that we want.
# https://gist.github.com/imneme/540829265469e673d045
# http://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html

# The MIT License (MIT)

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INIT_A = 0x43B0D7E5
MULT_A = 0x931E8875
INIT_B = 0x8B51F9DD
MULT_B = 0x58F38DED
MIX_MULT_L = 0xCA01F9DD
MIX_MULT_R = 0x4973F715
XSHIFT = np.dtype(np.uint32).itemsize * 8 // 2
MASK32 = 0xFFFFFFFF


def _generate_states(base_seed=0, worker_id=0):
    # init hash constant
    hash_const_A = INIT_A
    hash_const_B = INIT_B

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # init entropies with based_seed and worker_id and calculate pool
    entropies = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [hash(entropy) for entropy in entropies]

    # mix all bits together
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i != j:
                pool[j] = mix(pool[j], hash(pool[i]))

    states = []
    for p in pool:
        state = (p ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        state = (state * hash_const_B) & MASK32
        state = (state ^ (state >> XSHIFT)) & MASK32
        states.append(state)

    return states


def _worker_loop(
    dataset,
    dataset_kind,
    indices_queue,
    out_queue,
    done_event,
    auto_collate_batch,
    collate_fn,
    drop_last,
    init_fn,
    worker_id,
    num_workers,
    use_shared_memory,
    base_seed,
    shm_cache_size=0,
):
    try:
        # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
        # some shared memory objects may have been applied for but have not yet
        # been put into the inter-process Queue. This part of the object needs
        # to be cleaned up when the process ends.
        CleanupFuncRegistrar.register(_cleanup_mmap)

        # set signal handler
        core._set_process_signal_handler()

        core._set_max_memory_map_allocation_pool_size(shm_cache_size)

        # set different numpy seed for each worker
        try:
            import random

            import numpy as np
        except ImportError:
            pass
        else:
            seed = base_seed + worker_id
            random.seed(seed)
            paddle.seed(seed)
            np.random.seed(_generate_states(base_seed, worker_id))

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id,
            num_workers=num_workers,
            dataset=dataset,
            seed=base_seed,
        )

        init_exception = None
        try:
            if init_fn is not None:
                init_fn(worker_id)
            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collate_batch, collate_fn, drop_last
            )
        except:
            init_exception = _WorkerException(worker_id)

        iterator_drained = False
        parent_watch_dog = ParentWatchDog()

        while parent_watch_dog.is_alive():
            try:
                data = indices_queue.get(MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if isinstance(data, _ResumeIteration):
                out_queue.put((data, None, None))
                iterator_drained = False
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collate_batch, collate_fn, True
                )
                continue

            # None as poison piil, so worker event should be set
            if data is None:
                assert (
                    done_event.is_set() or iterator_drained
                ), "get None when worker done_event set"
                break
            # If worker done event is set but get still get data in
            # indices_queue, remaining data should be get and skipped.
            if done_event.is_set() or iterator_drained:
                continue

            idx, indices = data
            try:
                if init_exception is not None:
                    batch = init_exception
                    init_exception = None
                else:
                    # NOTE: GPU tensor operation is not supported in sub-process
                    #       but default device is GPU in paddle-gpu version, which
                    #       may copy CPU tensor to GPU even if users want to use
                    #       CPU tensor operation, so we add CPUPlace guard here
                    #       to make sure tensor will be operated only on CPU
                    with paddle.base.dygraph.guard(place=paddle.CPUPlace()):
                        batch = fetcher.fetch(indices)
            except Exception as e:
                if (
                    isinstance(e, StopIteration)
                    and dataset_kind == _DatasetKind.ITER
                ):
                    out_queue.put(_IterableDatasetStopIteration(worker_id))
                    iterator_drained = True
                else:
                    out_queue.put((idx, _WorkerException(worker_id), None))
            else:
                if isinstance(batch, _WorkerException):
                    out_queue.put((idx, batch, None))
                batch, structure = _flatten_batch(batch)
                if use_shared_memory:

                    def numpy2lodtensor(arr):
                        lodtensor = core.DenseTensor()
                        lodtensor.set(arr, core.CPUPlace())
                        return lodtensor

                    tensor_list = [
                        (
                            numpy2lodtensor(b)
                            if isinstance(b, np.ndarray)
                            else b.get_tensor()
                        )
                        for b in batch
                    ]
                    out_queue.put((idx, tensor_list, structure))
                else:
                    out_queue.put((idx, batch, structure))
    except KeyboardInterrupt:
        # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
        pass
    except:
        raise
    finally:
        if use_shared_memory:
            _cleanup_mmap()
    if done_event.is_set():
        out_queue.cancel_join_thread()
        out_queue.close()

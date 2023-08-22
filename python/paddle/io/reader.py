# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import multiprocessing

# NOTE: queue has a different name in python2 and python3
import sys
import time
import warnings

import paddle
from paddle.fluid.framework import logging

from ..fluid.framework import (
    _current_expected_place,
    _get_paddle_place,
    _get_paddle_place_list,
)
from ..framework import core, in_dynamic_mode
from .dataloader import BatchSampler, IterableDataset, Subset
from .dataloader.batch_sampler import _InfiniteIterableSampler
from .dataloader.dataloader_iter import (
    _DataLoaderIterMultiProcess,
    _DataLoaderIterSingleProcess,
    _DatasetKind,
)

# NOTE: [ avoid hanging & failed quickly ]
# These value is used in getting data from another process
QUEUE_GET_TIMEOUT = 60

USE_PINNED_MEMORY = None
# AutoTune Flags
USE_AUTOTUNE = False
TUNING_STEPS = 500


def set_autotune_config(use_autotune, tuning_steps=500):
    global USE_AUTOTUNE
    USE_AUTOTUNE = use_autotune
    global TUNING_STEPS
    TUNING_STEPS = tuning_steps


def use_pinned_memory(*args):
    global USE_PINNED_MEMORY
    if len(args) == 0:
        return USE_PINNED_MEMORY
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        USE_PINNED_MEMORY = args[0]


def _convert_places(places):
    if not isinstance(places, (list, tuple)):
        places = [places]

    ret = []
    for p in places:
        if not isinstance(p, core.Place):
            tmp = core.Place()
            tmp.set_place(p)
            p = tmp

        ret.append(p)
    return ret


class AuToTune:
    def __init__(self, loader):
        self.loader = loader
        self.max_num_worker = multiprocessing.cpu_count() / 2

    def __call__(self):
        # use default loader
        if (not USE_AUTOTUNE) or (not self.need_autotune()):
            return self.loader.num_workers

        # get autotune loader
        auto_tune_loader = self.get_autotune_loader()
        if auto_tune_loader is None:
            return self.loader.num_workers

        # pick the best num_workers
        auto_tune_start = time.time()
        logging.debug("========= DataLoader Auto Tune =========")
        logging.debug(
            "User config for DataLoader: " + str(self.loader.num_workers)
        )
        best_num_workers = 0
        min_cost = float("inf")
        logging.debug(
            "Tuning Range for num_workers: 0 ~ " + str(self.max_num_worker)
        )
        num_workers = 0
        while num_workers < self.max_num_worker:
            auto_tune_loader.num_workers = num_workers
            avg_cost = self.evaluate_reader_cost(auto_tune_loader)
            if min_cost * 0.75 > avg_cost:
                min_cost = avg_cost
                best_num_workers = num_workers
            else:
                update_num = self.is_best(
                    auto_tune_loader,
                    best_num_workers,
                    min_cost,
                    self.max_num_worker,
                )
                if update_num == best_num_workers:
                    break
                else:
                    best_num_workers = update_num
            logging.debug(
                "num_workers: "
                + str(num_workers)
                + " avg_cost: "
                + str(avg_cost)
            )
            num_workers += 2
        logging.info(
            "auto_tune dataLoader best_num_workers: " + str(best_num_workers)
        )
        logging.debug(
            "AutoTuning Cost for DataLoader: "
            + str(time.time() - auto_tune_start)
            + ' seconds'
        )

        # tune the default loader's num_workers
        return best_num_workers

    def need_autotune(self):
        if sys.platform == 'darwin' or sys.platform == 'win32':
            return False
        else:
            return True

    def get_sub_dataset(self, dataset, batch_size):
        num_samples = min(batch_size * TUNING_STEPS, len(dataset))
        sub_dataset = Subset(dataset, indices=list(range(num_samples)))
        return sub_dataset

    def get_autotune_loader(self):
        loader = copy.copy(self.loader)
        batch_size = self.loader.batch_sampler.batch_size
        if isinstance(
            self.loader.batch_sampler, paddle.io.DistributedBatchSampler
        ):
            dataset = self.loader.batch_sampler.dataset
            sub_dataset = self.get_sub_dataset(dataset, batch_size)
            loader.batch_sampler = paddle.io.DistributedBatchSampler(
                dataset=sub_dataset,
                batch_size=batch_size,
                num_replicas=self.loader.batch_sampler.nranks,
                rank=self.loader.batch_sampler.local_rank,
                shuffle=self.loader.batch_sampler.shuffle,
                drop_last=self.loader.batch_sampler.drop_last,
            )
        elif isinstance(self.loader.batch_sampler, paddle.io.BatchSampler):
            dataset = self.loader.batch_sampler.sampler.data_source
            sub_dataset = self.get_sub_dataset(dataset, batch_size)
            loader.batch_sampler = paddle.io.BatchSampler(
                dataset=sub_dataset,
                batch_size=batch_size,
                drop_last=self.loader.batch_sampler.drop_last,
            )
        else:
            loader = None
        return loader

    def evaluate_reader_cost(self, reader):
        costs = []
        avg_cost = 0
        start = time.time()
        for i, data in enumerate(reader):
            costs.append(time.time() - start)
            start = time.time()
        if len(costs) > 2:
            avg_cost = sum(costs[2:]) / len(costs[2:])
        else:
            avg_cost = sum(costs[0:]) / len(costs[0:])
        return avg_cost

    def is_best(self, reader, best_workers, best_time, num_work_boundary):
        step = 0
        num_workers = best_workers + 1
        boundary = 1
        while num_workers < num_work_boundary and step < 5:
            self.loader.num_workers = num_workers
            time = self.evaluate_reader_cost(reader)
            logging.debug(
                "for back num_workers: "
                + str(num_workers)
                + " avg_cost: "
                + str(time)
            )
            step += 1
            if time < best_time * 0.70 * boundary:
                return num_workers
            else:
                num_workers += 1
            boundary *= 0.80
        return best_workers


class DataLoader:
    """
    DataLoader prodives an iterator which iterates given dataset
    once by the batch_sampler.

    DataLoader supports single-process and multi-prcess data loading,
    multi-process workers will be used to load data asynchronously if
    :attr:`num_workers` is set as a positive number.

    DataLoader supports map-style dataset and iterable-style dataset.

    For map-style datast(can get a sample from dataset with a given
    index), please see :code:`paddle.io.Dataset`.

    For iterable-style datast(get samples from dataset iteratively,
    like a Python iterator), please see :code:`paddle.io.IterableDataset`.

    For :code:`batch_sampler` please see :code:`paddle.io.BatchSampler`

    .. note::
        GPU tensor operation is not supported in subprocess currently,
        please don't use GPU tensor operations in pipeline which will
        be performed in subprocess, such as dataset transforms, collte_fn,
        etc. Numpy array and CPU tensor operation is supported.

    **Disable automatic batching**

    In certain cases such as some NLP tasks, instead of automatic batching,
    handling batching manually in dataset is needed by users. For these
    cases, automatic batching is disabled if both :attr:`batch_size` and
    :attr:`batch_sampler` is set as None, each data got from :attr:`dataset`
    should be batched data and will be processed with function define by
    :attr:`collate_fn` or :attr:`default_collate_fn`.


    .. note::
        When automatic batching is disabled, :attr:`default_collate_fn` will
        do nothing to data from dataset.


    Args:
        dataset(Dataset): the dataset to load data from, should be an
            instance of subclass of :code:`paddle.io.Dataset` or
            :code:`paddle.io.IterableDataset`.
        feed_list (list(Tensor)|tuple(Tensor), optional): feed Tensor list.
            The Tensors should be created by :code:`paddle.static.data()`.
            :attr:`feed_list` must be set if :attr:`return_list` is
            False. Default None.
        places(list(Place)|tuple(Place)|list(str), optional): a list of Place,
            to put data onto, :attr:`places` can be None, if
            :attr:`places` is None, default place(CPUPlace or CUDAPlace(0))
            will be used. Default None. If ``places`` is list of string,
            the string in the list can be ``cpu``, ``gpu:x`` and ``gpu_pinned``,
            where ``x`` is the index of the GPUs.
        return_list (bool, optional): whether the return value on each device is
            presented as a list. If :attr:`return_list=False`, the return
            value on each device would be a dict of str -> Tensor, where
            the key of the dict is the name of each fed Tensors. If
            :attr:`return_list=True`, the return value on each device would
            be a list(Tensor). :attr:`return_list` can only be True
            in dynamic graph mode. Default True.
        batch_sampler(BatchSampler, optional): an instance of `paddle.io.BatchSampler`
            to generate batch indices to draw samples from :attr:`dataset`
            and combine a batch. Default None.
        batch_size(int|None, optional): sample number in a mini-batch, a substitution
            parameter for :attr:`batch_sampler`, if :attr:`batch_sampler`
            is not set, a default `paddle.io.BatchSampler` will be used
            and initialize by :attr:`batch_size`, :attr:`shuffle` and
            :attr:`drop_last`. Default 1.
        shuffle(bool, optional): whther to shuffle indices order before genrate
            batch indices, a substitution parameter for :attr:`batch_sampler`
            see :attr:`batch_size`. Default False.
        drop_last(bool, optional): whether drop the last incomplete batch dataset size
            is not divisible by the batch size, a substitution parameter
            for :attr:`batch_sampler`, see :attr:`batch_size`. Default False
        collate_fn(callable, optional): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`). Default None
        num_workers(int, optional): the number of subprocess to load data, 0 for no
            subprocess used and loading data in main process. Default 0
        use_buffer_reader (bool, optional): whether to use bufferred reader.
            If use_buffer_reader=True, the DataLoader would prefetch
            batch data asynchronously, so it would speed up data feeding
            and occupies a little more CPU or GPU memory, i.e., the memory
            of one batch input data. Default True.
        prefetch_factor (int, optional): Number of batch data the DataLoader would prefetch
            if use_buffer_reader=True. Default 2.
        use_shared_memory (bool, optional): whether to use shared memory to speed up
            putting data into inter-process queue, set :attr:`use_shared_memory`
            as True only when the shared memory space on your machine(e.g.
            space of '/dev/shm' on Linux operating sysytem) is large enough.
            Shared memory will only be enabled in multi-process mode(num_workers
            > 0). Default True.
        timeout(int, optional): the timeout value for getting data form output queue
            of subprocesses. Default 0.
        worker_init_fn(callable, optional): init function which will be called with
            worker id on each subproces starting if not set as None. Default
            None.

    Returns:
        DataLoader: an iterable object for data iterating, each elemnet of the generated data is a Tensor.

    Examples:

        .. code-block:: python

            import numpy as np

            import paddle
            import paddle.nn as nn
            import paddle.nn.functional as F
            from paddle.io import Dataset, BatchSampler, DataLoader

            BATCH_NUM = 20
            BATCH_SIZE = 16
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

            class SimpleNet(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                def forward(self, image, label=None):
                    return self.fc(image)

            simple_net = SimpleNet()
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                      parameters=simple_net.parameters())

            loader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                num_workers=2)

            for e in range(EPOCH_NUM):
                for i, (image, label) in enumerate(loader()):
                    out = simple_net(image)
                    loss = F.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    simple_net.clear_gradients()
                    print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))


    .. note::
        For reading iterable dataset with multiprocess Dataloader,
        please see :code:`paddle.io.IterableDataset`

    """

    def __init__(
        self,
        dataset,
        feed_list=None,
        places=None,
        return_list=True,
        batch_sampler=None,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        prefetch_factor=2,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,
    ):
        self.return_list = return_list
        self.collate_fn = collate_fn
        self.use_buffer_reader = use_buffer_reader
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn

        self.dataset = dataset

        if not return_list and not in_dynamic_mode():
            assert (
                feed_list is not None
            ), "feed_list should be set when return_list=False"
        self.feed_list = feed_list

        if places is None:
            places = _current_expected_place()
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        self.places = _convert_places(places)

        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 0 and (
            sys.platform == 'darwin' or sys.platform == 'win32'
        ):
            warnings.warn(
                "DataLoader with multi-process mode is not supported on MacOs and Windows currently."
                " Please use signle-process mode with num_workers = 0 instead"
            )
            num_workers = 0
        self.num_workers = num_workers

        assert prefetch_factor > 0, "prefetch_factor should be a positive value"

        self.use_shared_memory = use_shared_memory
        if use_shared_memory and num_workers == 0:
            self.use_shared_memory = False

        assert timeout >= 0, "timeout should be a non-negative value"
        self.timeout = timeout

        if isinstance(dataset, IterableDataset):
            self.dataset_kind = _DatasetKind.ITER
            if shuffle:
                raise ValueError(
                    "IterableDataset not support shuffle, but got shuffle={}".format(
                        shuffle
                    )
                )
            if batch_sampler is not None:
                raise ValueError(
                    "IterableDataset expect unspecified batch_sampler"
                )
        else:
            self.dataset_kind = _DatasetKind.MAP

        if batch_sampler is not None:
            assert batch_size == 1 and not shuffle and not drop_last, (
                "batch_size/shuffle/drop_last should not be set when "
                "batch_sampler is given"
            )
            self.batch_sampler = batch_sampler
            self.batch_size = None
        elif batch_size is None:
            self.batch_sampler = None
            self.batch_size = None
        else:
            assert batch_size > 0, (
                "batch_size should be None or a positive value when "
                "batch_sampler is not given"
            )
            self.batch_size = batch_size
            if isinstance(dataset, IterableDataset):
                self.batch_sampler = _InfiniteIterableSampler(
                    dataset, batch_size
                )
            else:
                self.batch_sampler = BatchSampler(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )

        self.drop_last = drop_last
        self.auto_collate_batch = self.batch_sampler is not None

        self.pin_memory = False
        if in_dynamic_mode():
            self.pin_memory = (
                True if use_pinned_memory() is None else use_pinned_memory()
            )

        self._persistent_workers = persistent_workers
        self._iterator = None
        self.num_workers = AuToTune(self).__call__()

    def __len__(self):
        if self.dataset_kind == _DatasetKind.ITER:
            raise ValueError("length of IterableDataset not supported")
        else:
            if self.auto_collate_batch:
                return len(self.batch_sampler)
            else:
                return len(self.dataset)

    def __iter__(self):
        if self.num_workers == 0:
            return _DataLoaderIterSingleProcess(self)
        elif self._persistent_workers:
            if self._iterator is None:
                self._iterator = _DataLoaderIterMultiProcess(self)
            else:
                self._iterator._reset()
            return self._iterator
        else:
            return _DataLoaderIterMultiProcess(self)

    def __call__(self):
        return self.__iter__()

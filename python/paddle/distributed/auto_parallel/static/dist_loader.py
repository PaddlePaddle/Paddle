# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import abc

import numpy as np

import paddle
from paddle.io import BatchSampler, IterableDataset
from paddle.io.dataloader.batch_sampler import (
    DistributedBatchSampler,
    _InfiniteIterableSampler,
)
from paddle.io.dataloader.dataloader_iter import (
    _DatasetKind,
    default_collate_fn,
    default_convert_fn,
)


class DistributedDataLoaderBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class DistributedDataLoaderFromGenerator(DistributedDataLoaderBase):
    def __init__(
        self,
        dataset,
        feed_list=None,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=False,
        use_multiprocess=False,
        drop_last=True,
        places=None,
        batch_size=1,
        epochs=1,
        steps_per_epoch=None,
        collate_fn=None,
        split_data=True,
        data_parallel_world_size=[],
        data_parallel_rank=[],
        acc_steps=1,
    ):
        self.dataset = dataset
        self.feed_list = feed_list
        self.capacity = capacity
        self.use_double_buffer = use_double_buffer
        self.iterable = iterable
        self.return_list = return_list
        self.use_multiprocess = use_multiprocess
        self.drop_last = drop_last
        self.places = places
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.collate_fn = collate_fn
        self.split_data = split_data
        assert len(data_parallel_world_size) == len(feed_list)
        assert len(data_parallel_rank) == len(feed_list)
        self.dp_world_sizes = data_parallel_world_size
        self.dp_ranks = data_parallel_rank
        self.acc_steps = acc_steps

        if isinstance(dataset, IterableDataset):
            self.dataset_kind = _DatasetKind.ITER
        else:
            self.dataset_kind = _DatasetKind.MAP

        if self.batch_size is None:
            self.batch_sampler = None
        else:
            if isinstance(dataset, IterableDataset):
                self.batch_sampler = _InfiniteIterableSampler(
                    dataset, batch_size
                )
            else:
                self.batch_sampler = BatchSampler(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=drop_last,
                )

        self.auto_collate_batch = self.batch_sampler is not None
        self.sampler_iter = iter(self.index_sampler)

        if self.auto_collate_batch:
            self.collate_fn = collate_fn or default_collate_fn
        else:
            self.collate_fn = collate_fn or default_convert_fn

        self.dataset_fetcher = _DatasetKind.create_fetcher(
            self.dataset_kind,
            self.dataset,
            self.auto_collate_batch,
            self.collate_fn,
            self.drop_last,
        )

        self._steps = self._infer_steps()
        self._inner_dataloader = self._create_inner_dataloader()

    def __iter__(self):
        self._cur_step = 0
        self._inner_dataloader.start()
        return self

    def __next__(self):
        if not self._steps:
            self._cur_step += 1
            return None
        elif self._cur_step < self._steps:
            self._cur_step += 1
            return None
        else:
            self._inner_dataloader.reset()
            self.sampler_iter = iter(self.index_sampler)
            raise StopIteration

    def _infer_steps(self):
        if isinstance(self.steps_per_epoch, int) and self.steps_per_epoch > 0:
            return self.steps_per_epoch
        try:
            if isinstance(self.dataset, IterableDataset):
                steps_per_epoch = None
            elif self.batch_size is None:
                steps_per_epoch = len(self.dataset) // self.acc_steps
            else:
                steps_per_epoch = (
                    len(self.dataset) // self.batch_size // self.acc_steps
                )
        except:
            raise ValueError(
                "Please set `steps_per_epoch` or implement `__len__` method in dataset class."
            )
        return steps_per_epoch

    @property
    def index_sampler(self):
        if self.auto_collate_batch:
            return self.batch_sampler
        else:
            if self.dataset_kind == _DatasetKind.MAP:
                return list(range(len(self.dataset)))
            else:
                return _InfiniteIterableSampler(self.dataset, 1)

    def _create_inner_dataloader(self):
        def data_generator():
            while True:
                try:
                    indices = next(self.sampler_iter)
                    batch = self.dataset_fetcher.fetch(indices)
                    if batch is None:
                        break
                except StopIteration:
                    self.dataset_fetcher = _DatasetKind.create_fetcher(
                        self.dataset_kind,
                        self.dataset,
                        self.auto_collate_batch,
                        self.collate_fn,
                        self.drop_last,
                    )
                    break

                partial_data = []
                for i, d in enumerate(batch):
                    array = np.array(d)
                    if not self.split_data:
                        partial_data.append(array)
                        continue

                    batch_size = array.shape[0]
                    assert (
                        batch_size % self.dp_world_sizes[i] == 0
                    ), f"batch_size [{str(batch_size)}] is not divisible by dp_world_size [{str(self.dp_world_sizes[i])}]"
                    partial_data.append(
                        np.split(array, self.dp_world_sizes[i])[
                            self.dp_ranks[i]
                        ]
                    )

                yield partial_data

        dataloader = paddle.base.io.DataLoader.from_generator(
            feed_list=self.feed_list,
            capacity=self.capacity,
            use_double_buffer=self.use_double_buffer,
            # iterable=self.iterable,
            iterable=False,
            return_list=self.return_list,
            use_multiprocess=self.use_multiprocess,
            drop_last=self.drop_last,
        )
        dataloader.set_batch_generator(data_generator, self.places)

        return dataloader


class DistributedDataLoader(DistributedDataLoaderBase):
    def __init__(
        self,
        dataset,
        feed_list=None,
        places=None,
        return_list=True,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        steps_per_epoch=None,
        split_data=True,
        data_parallel_world_size=[],
        data_parallel_rank=[],
    ):
        self.dataset = dataset
        self.feed_list = feed_list
        self.return_list = return_list
        self.places = places
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.use_buffer_reader = use_buffer_reader
        self.use_shared_memory = use_shared_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dp_world_sizes = data_parallel_world_size
        self.dp_ranks = data_parallel_rank
        self.split_data = split_data

        if self.batch_size is None:
            self.batch_sampler = None
        else:
            self.batch_sampler = DistributedBatchSampler(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_replicas=self.dp_world_sizes[0],
                rank=self.dp_ranks[0],
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )

        self._dataloader = paddle.io.DataLoader(
            self.dataset,
            feed_list=self.feed_list,
            places=self.places,
            return_list=self.return_list,
            batch_sampler=self.batch_sampler,
            batch_size=1 if self.batch_sampler else self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            use_buffer_reader=self.use_buffer_reader,
            use_shared_memory=self.use_shared_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
        )

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self._dataloader.__iter__()

    def __call__(self):
        return self._dataloader.__iter__()

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
from functools import wraps

import paddle
from .utils import to_list
from paddle.fluid.layers.utils import flatten
from paddle.io import DataLoader, BatchSampler, IterableDataset
from paddle.fluid.dataloader.batch_sampler import _InfiniteIterableSampler
from paddle.fluid.dataloader.dataloader_iter import _DatasetKind, default_collate_fn, default_convert_fn


class DistributedDataLoader(metaclass=abc.ABCMeta):

    def __init__(self, dataset, batch_size=1, epochs=1, drop_last=False):
        if isinstance(dataset, IterableDataset):
            self.dataset_kind = _DatasetKind.ITER
        else:
            self.dataset_kind = _DatasetKind.MAP

        self.dataset = dataset
        self.epochs = epochs
        self.drop_last = drop_last

        if batch_size is None:
            self.batch_size = None
            self.batch_sampler = None
        else:
            self.batch_size = batch_size
            if isinstance(dataset, IterableDataset):
                self.batch_sampler = _InfiniteIterableSampler(
                    dataset, batch_size)
            else:
                self.batch_sampler = BatchSampler(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=drop_last)

        self.auto_collate_batch = self.batch_sampler is not None
        self.sampler_iter = iter(self.index_sampler)

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError

    @property
    def index_sampler(self):
        if self.auto_collate_batch:
            return self.batch_sampler
        else:
            if self.dataset_kind == _DatasetKind.MAP:
                return list(range(len(self.dataset)))
            else:
                return _InfiniteIterableSampler(self.dataset, 1)


class NonIterableGeneratorLoader(DistributedDataLoader):

    def __init__(self,
                 dataset,
                 feed_list,
                 places,
                 batch_size=1,
                 epochs=1,
                 steps_per_epoch=None,
                 collate_fn=None,
                 data_parallel_world_size=[],
                 data_parallel_rank=[],
                 drop_last=False,
                 split_data=True):
        self.feed_list = feed_list
        self.places = places
        self.steps_per_epoch = steps_per_epoch

        assert len(data_parallel_world_size) == len(feed_list)
        assert len(data_parallel_rank) == len(feed_list)
        self.dp_world_sizes = data_parallel_world_size
        self.dp_ranks = data_parallel_rank
        self.split_data = split_data

        super(NonIterableGeneratorLoader,
              self).__init__(dataset, batch_size, epochs, drop_last)

        if self.auto_collate_batch:
            self.collate_fn = collate_fn or default_collate_fn
        else:
            self.collate_fn = collate_fn or default_convert_fn
        self.dataset_fetcher = _DatasetKind.create_fetcher(
            self.dataset_kind, self.dataset, self.auto_collate_batch,
            self.collate_fn, self.drop_last)

        self._steps = self._infer_steps()
        self._inner_dataloader = self._create_inner_dataloader()

    def __iter__(self):
        self._cur_step = 0
        self._inner_dataloader.start()
        return self

    def __next__(self):
        if not self._steps:
            self._cur_step += 1
        elif self._cur_step < self._steps:
            self._cur_step += 1
        else:
            self._inner_dataloader.reset()
            self.sampler_iter = iter(self.index_sampler)
            raise StopIteration

    def _infer_steps(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch
        try:
            if isinstance(self.dataset, IterableDataset):
                steps_per_epoch = None
            elif self.batch_size is None:
                steps_per_epoch = len(self.dataset)
            else:
                steps_per_epoch = len(self.dataset) // self.batch_size
        except:
            raise ValueError(
                "Pleace set `steps_per_epoch` or implement `__len__` methond in dataset class."
            )
        return steps_per_epoch

    def _create_inner_dataloader(self):

        def data_generator():
            while True:
                try:
                    indices = next(self.sampler_iter)
                    batch = self.dataset_fetcher.fetch(indices)
                    if batch is None: break
                except StopIteration:
                    self.dataset_fetcher = _DatasetKind.create_fetcher(
                        self.dataset_kind, self.dataset,
                        self.auto_collate_batch, self.collate_fn,
                        self.drop_last)
                    break

                partial_data = []
                for i, d in enumerate(batch):
                    array = np.array(d)
                    if not self.split_data:
                        partial_data.append(array)
                        continue

                    batch_size = array.shape[0]
                    assert batch_size % self.dp_world_sizes[i] == 0, \
                        "batch_size [{}] is not divisible by dp_world_size [{}]".format(str(batch_size), str(self.dp_world_sizes[i]))
                    partial_data.append(
                        np.split(array,
                                 self.dp_world_sizes[i])[self.dp_ranks[i]])

                yield partial_data

        dataloader = paddle.fluid.io.DataLoader.from_generator(
            feed_list=self.feed_list, capacity=70, iterable=False)
        dataloader.set_batch_generator(data_generator, self.places)

        return dataloader

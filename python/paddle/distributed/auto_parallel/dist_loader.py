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
from .utils import to_list
from paddle.fluid.layers.utils import flatten
from paddle.io import DataLoader, BatchSampler, IterableDataset
from paddle.fluid.dataloader.dataloader_iter import _DatasetKind, default_collate_fn, default_convert_fn


class DistributedDataLoader(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 epochs=1,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 drop_last=False):
        if isinstance(dataset, IterableDataset):
            raise TypeError("IterableDataset is not supported.")
        else:
            self.dataset_kind = _DatasetKind.MAP

        self.dataset = dataset
        self.epochs = epochs
        self.drop_lost = drop_last

        if batch_size is None:
            self.batch_size = None
            self.batch_sampler = None
        else:
            if data_parallel_world_size is not None:
                assert batch_size % data_parallel_world_size == 0, \
                    "'batch_size' must be divisible by data parallel size"
            self.batch_size = batch_size
            self.batch_sampler = BatchSampler(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=drop_last)

        self.auto_collate_batch = self.batch_sampler is not None
        self.sampler_iter = iter(self.index_sampler)
        self.dp_world_size = 1 if data_parallel_world_size is None else data_parallel_world_size
        self.dp_rank = 0 if data_parallel_rank is None else data_parallel_rank

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
                raise TypeError("Only support datasets in map-style.")


class NonIterableGeneratorLoader(DistributedDataLoader):

    def __init__(self,
                 dataset,
                 feed_list,
                 places,
                 batch_size=1,
                 epochs=1,
                 steps_per_epoch=None,
                 collate_fn=None,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 drop_last=False,
                 split_data=True):
        self.feed_list = feed_list
        self.places = places
        self.steps_per_epoch = steps_per_epoch
        self.split_data = split_data

        super(NonIterableGeneratorLoader,
              self).__init__(dataset, batch_size, epochs,
                             data_parallel_world_size, data_parallel_rank,
                             drop_last)

        if self.auto_collate_batch:
            self.collate_fn = collate_fn or default_collate_fn
        else:
            self.collate_fn = collate_fn or default_convert_fn
        self.dataset_fetcher = _DatasetKind.create_fetcher(
            self.dataset_kind, self.dataset, self.auto_collate_batch,
            self.collate_fn, self.drop_lost)

        self._steps = self._infer_steps()
        self._inner_dataloader = self._create_inner_dataloader()

    def __iter__(self):
        self._cur_step = 0
        self._inner_dataloader.start()
        return self

    def __next__(self):
        if self._cur_step < self._steps:
            self._cur_step += 1
        else:
            self._inner_dataloader.reset()
            raise StopIteration

    def _infer_steps(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch
        try:
            if self.batch_size is None:
                steps_per_epoch = len(self.dataset)
            else:
                steps_per_epoch = len(self.dataset) // self.batch_size
        except:
            raise ValueError(
                "Pleace set `steps_per_epoch` or implement `__len__` methond in dataset class."
            )
        return steps_per_epoch

    def _create_inner_dataloader(self):

        def sample_data_generator():
            for indices in self.sampler_iter:
                assert len(indices) % self.dp_world_size == 0, \
                    "Please set batch_size to be divisible by data parallel size"
                n = len(indices) // self.dp_world_size
                cur_indices = [
                    indices[i:i + n] for i in range(0, len(indices), n)
                ]
                batch = self.dataset_fetcher.fetch(cur_indices[self.dp_rank])
                yield batch[:len(self.feed_list)]

        def batch_data_generator():
            for indices in self.sampler_iter:
                partial_data = []
                batch = self.dataset_fetcher.fetch(indices)

                for data in batch:

                    if not self.split_data:
                        # array = np.array(data)
                        partial_data.append(data)
                        continue

                    assert data.shape[0] % self.dp_world_size == 0, \
                        "Please padding dataset's batch_size to be divisible by data parallel size"
                    partial_data.append(
                        np.split(data, self.dp_world_size)[self.dp_rank])
                yield partial_data[:len(self.feed_list)]

        dataloader = paddle.fluid.io.DataLoader.from_generator(
            feed_list=self.feed_list, capacity=70, iterable=False)
        if self.batch_size is not None:
            dataloader.set_batch_generator(sample_data_generator, self.places)
        else:
            dataloader.set_batch_generator(batch_data_generator, self.places)

        return dataloader

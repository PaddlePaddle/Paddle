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
from paddle.io import DataLoader, DistributedBatchSampler


class DistributedDataLoader(metaclass=abc.ABCMeta):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 epochs=1,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = data_parallel_rank
        self.drop_lost = drop_last
        if data_parallel_world_size is not None:
            assert batch_size % data_parallel_world_size == 0

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


class NonIterableGeneratorLoader(DistributedDataLoader):
    def __init__(self,
                 dataset,
                 feed_list,
                 places,
                 batch_size=1,
                 epochs=1,
                 steps_per_epoch=1000,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 drop_last=False):
        self.feed_list = feed_list
        self.places = places
        self.steps_per_epoch = steps_per_epoch
        super(NonIterableGeneratorLoader, self).__init__(
            dataset, batch_size, epochs, data_parallel_world_size,
            data_parallel_rank, drop_last)
        self._inner_dataloader = self._create_inner_dataloader()

    def __iter__(self):
        self._cur_step = 0
        self._inner_dataloader.start()
        return self

    def __next__(self):
        if self._cur_step < self.steps_per_epoch:
            self._cur_step += 1
        else:
            self._inner_dataloader.reset()
            raise StopIteration

    def _create_inner_dataloader(self):
        def data_generator():
            batch_data = None
            for step, data in enumerate(self.dataset):
                if batch_data is None:
                    batch_data = [[] for i in range(len(data))]
                for idx, data_item in enumerate(data):
                    batch_data[idx].append(np.array(data_item))
                if (step + 1) % self.batch_size == 0:
                    yield batch_data[0], batch_data[1]
                    batch_data = None

        dataloader = paddle.fluid.io.DataLoader.from_generator(
            feed_list=self.feed_list, capacity=70, iterable=False)
        dataloader.set_batch_generator(data_generator, self.places)
        return dataloader

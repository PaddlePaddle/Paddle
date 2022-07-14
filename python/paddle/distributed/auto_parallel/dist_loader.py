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
        if data_parallel_world_size is not None and batch_size is not None:
            for dp_world_size in data_parallel_world_size:
                assert batch_size % dp_world_size == 0, \
                    "batch_size must be divisible by dp_world_size value {}".format(str(dp_world_size))

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
                 steps_per_epoch=None,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 drop_last=False):
        self.feed_list = feed_list
        self.places = places
        self.steps_per_epoch = steps_per_epoch

        super(NonIterableGeneratorLoader,
              self).__init__(dataset, batch_size, epochs,
                             data_parallel_world_size, data_parallel_rank,
                             drop_last)
        self._inner_dataloader = self._create_inner_dataloader()
        self._steps = self._infer_steps()

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
            batch_data = None
            for step, data in enumerate(self.dataset):
                data = flatten(data)
                if batch_data is None:
                    batch_data = [[] for i in range(len(data))]
                for idx in range(len(data)):
                    batch_data[idx].append(data[idx])
                if (step + 1) % self.batch_size == 0:
                    partial_data = []
                    for i, d in enumerate(batch_data[:len(self.feed_list)]):
                        array = np.array(d)
                        partial_data.append(
                            np.split(array,
                                     self.dp_world_size[i])[self.dp_rank[i]])
                    yield partial_data
                    batch_data = None

        def batch_data_generator():
            for data in self.dataset:
                data = flatten(data)
                partial_data = []
                for d in data:
                    assert d.shape[0] % self.dp_world_size == 0, \
                        "Please padding dataset with data parallel size"
                    partial_data.append(
                        np.split(d, self.dp_world_size)[self.dp_rank])
                yield partial_data[:len(self.feed_list)]

        self.dp_world_size = [
            1 for _ in range(len(self.feed_list))
        ] if self.data_parallel_world_size is None else self.data_parallel_world_size
        self.dp_rank = [
            0 for _ in range(len(self.feed_list))
        ] if self.data_parallel_rank is None else self.data_parallel_rank
        dataloader = paddle.fluid.io.DataLoader.from_generator(
            feed_list=self.feed_list, capacity=70, iterable=False)
        if self.batch_size is not None:
            dataloader.set_batch_generator(sample_data_generator, self.places)
        else:
            dataloader.set_batch_generator(batch_data_generator, self.places)

        return dataloader

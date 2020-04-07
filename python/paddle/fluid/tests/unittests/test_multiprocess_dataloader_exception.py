# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import os
import sys
import six
import time
import unittest
import multiprocessing
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.io import Dataset, BatchSampler, DataLoader
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable


class RandomDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.sample_num


class TestDataLoaderSetXXXException(unittest.TestCase):
    def test_main(self):
        place = fluid.cpu_places()[0]
        with fluid.dygraph.guard(place):
            dataset = RandomDataset(800)
            dataloader = DataLoader(dataset, places=place)

            try:
                dataloader.set_sample_generator()
                self.assertTrue(False)
            except:
                pass

            try:
                dataloader.set_sample_list_generator()
                self.assertTrue(False)
            except:
                pass

            try:
                dataloader.set_batch_generator()
                self.assertTrue(False)
            except:
                pass


class TestDataLoaderAssert(unittest.TestCase):
    def test_main(self):
        place = fluid.cpu_places()[0]
        with fluid.dygraph.guard(place):
            dataset = RandomDataset(800)
            batch_sampler = BatchSampler(dataset=dataset)

            # dataset is not instance of Dataset
            try:
                loader = DataLoader(dataset=batch_sampler, places=place)
                self.assertTrue(False)
            except AssertionError:
                pass

            # places is None
            try:
                loader = DataLoader(dataset=dataset, places=None)
                self.assertTrue(False)
            except AssertionError:
                pass

            # num_workers < 0
            try:
                loader = DataLoader(
                    dataset=dataset, places=place, num_workers=-1)
                self.assertTrue(False)
            except AssertionError:
                pass

            # timeout < 0
            try:
                loader = DataLoader(dataset=dataset, places=place, timeout=-1)
                self.assertTrue(False)
            except AssertionError:
                pass

            # batch_sampler is not instance of BatchSampler
            try:
                loader = DataLoader(
                    dataset=dataset, places=place, batch_sampler=dataset)
                self.assertTrue(False)
            except AssertionError:
                pass

            # set batch_sampler and shuffle/batch_size/drop_last
            try:
                loader = DataLoader(
                    dataset=dataset,
                    places=place,
                    batch_sampler=batch_sampler,
                    shuffle=True,
                    drop_last=True)
                self.assertTrue(False)
            except AssertionError:
                pass

            # set batch_sampler correctly
            try:
                loader = DataLoader(
                    dataset=dataset, places=place, batch_sampler=batch_sampler)
                self.assertTrue(True)
            except AssertionError:
                self.assertTrue(False)


# CI Converage cannot record stub in subprocess,
# HACK a _worker_loop in main process call here
class TestDataLoaderWorkerLoop(unittest.TestCase):
    def run_main(self, use_shared_memory=True):
        try:
            place = fluid.cpu_places()[0]
            with fluid.dygraph.guard(place):
                dataset = RandomDataset(800)
                loader = DataLoader(
                    dataset,
                    num_workers=1,
                    places=place,
                    use_shared_memory=use_shared_memory)
                loader = iter(loader)
                indices_queue = multiprocessing.Queue()
                indices_queue.put([0, 1])
                indices_queue.put([2, 3])
                indices_queue.put(None)
                loader._worker_loop(loader._dataset, indices_queue,
                                    loader._data_queue,
                                    loader._workers_done_event, None, None, 0)
                self.assertTrue(False)
        except AssertionError:
            pass
        except:
            self.assertTrue(False)

    def test_main(self):
        for use_shared_memory in [True, False]:
            self.run_main(use_shared_memory)


if __name__ == '__main__':
    unittest.main()

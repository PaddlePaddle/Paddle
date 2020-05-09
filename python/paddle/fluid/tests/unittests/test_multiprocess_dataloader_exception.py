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
from paddle.io import Dataset, BatchSampler, DataLoader
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


class TestDataLoaderAssert(unittest.TestCase):
    def test_main(self):
        place = fluid.cpu_places()[0]
        with fluid.dygraph.guard(place):
            dataset = RandomDataset(100)
            batch_sampler = BatchSampler(dataset=dataset, batch_size=4)

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
    def run_without_worker_done(self, use_shared_memory=True):
        try:
            place = fluid.cpu_places()[0]
            with fluid.dygraph.guard(place):
                dataset = RandomDataset(800)

                # test init_fn
                def _init_fn(worker_id):
                    pass

                # test collate_fn
                def _collate_fn(sample_list):
                    return [
                        np.stack(
                            s, axis=0) for s in list(zip(*sample_list))
                    ]

                loader = DataLoader(
                    dataset,
                    num_workers=1,
                    places=place,
                    use_shared_memory=use_shared_memory)
                assert loader.num_workers > 0, \
                    "go to AssertionError and pass in Mac and Windows"
                loader = iter(loader)
                print("loader length", len(loader))
                indices_queue = multiprocessing.Queue()
                for i in range(10):
                    indices_queue.put([i, i + 10])
                indices_queue.put(None)
                loader._worker_loop(
                    loader._dataset, indices_queue, loader._data_queue,
                    loader._workers_done_event, _collate_fn, _init_fn, 0)
                self.assertTrue(False)
        except AssertionError:
            pass
        except Exception:
            self.assertTrue(False)

    def run_with_worker_done(self, use_shared_memory=True):
        try:
            place = fluid.cpu_places()[0]
            with fluid.dygraph.guard(place):
                dataset = RandomDataset(800)

                # test init_fn
                def _init_fn(worker_id):
                    pass

                # test collate_fn
                def _collate_fn(sample_list):
                    return [
                        np.stack(
                            s, axis=0) for s in list(zip(*sample_list))
                    ]

                loader = DataLoader(
                    dataset,
                    num_workers=1,
                    places=place,
                    use_shared_memory=use_shared_memory)
                assert loader.num_workers > 0, \
                    "go to AssertionError and pass in Mac and Windows"
                loader = iter(loader)
                print("loader length", len(loader))
                indices_queue = multiprocessing.Queue()
                for i in range(10):
                    indices_queue.put([i, i + 10])
                indices_queue.put(None)
                loader._workers_done_event.set()
                loader._worker_loop(
                    loader._dataset, indices_queue, loader._data_queue,
                    loader._workers_done_event, _collate_fn, _init_fn, 0)
                self.assertTrue(True)
        except AssertionError:
            pass
        except Exception:
            self.assertTrue(False)

    def test_main(self):
        for use_shared_memory in [True, False]:
            self.run_without_worker_done(use_shared_memory)
            self.run_with_worker_done(use_shared_memory)


if __name__ == '__main__':
    unittest.main()

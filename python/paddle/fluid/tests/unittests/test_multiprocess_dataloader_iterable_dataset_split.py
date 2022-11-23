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

import math
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.io import DataLoader, IterableDataset, get_worker_info


class RangeIterableDatasetSplit(IterableDataset):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil(
<<<<<<< HEAD
                    (self.end - self.start) / float(worker_info.num_workers)))
=======
                    (self.end - self.start) / float(worker_info.num_workers)
                )
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for i in range(iter_start, iter_end):
            yield np.array([i])


class TestDynamicDataLoaderIterSplit(unittest.TestCase):

    def test_main(self):
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = RangeIterableDatasetSplit(0, 10)
<<<<<<< HEAD
            dataloader = DataLoader(dataset,
                                    places=place,
                                    num_workers=2,
                                    batch_size=1,
                                    drop_last=True)
=======
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=2,
                batch_size=1,
                drop_last=True,
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

            rets = []
            for d in dataloader:
                rets.append(d.numpy()[0][0])

            assert tuple(sorted(rets)) == tuple(range(0, 10))


class RangeIterableDataset(IterableDataset):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield np.array([i])


class TestDynamicDataLoaderIterInitFuncSplit(unittest.TestCase):

    def test_main(self):
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = RangeIterableDataset(0, 10)

            def worker_spliter(worker_id):
                worker_info = get_worker_info()

                dataset = worker_info.dataset
                start = dataset.start
                end = dataset.end
                num_per_worker = int(
                    math.ceil((end - start) / float(worker_info.num_workers))
                )

                worker_id = worker_info.id
                dataset.start = start + worker_id * num_per_worker
                dataset.end = min(dataset.start + num_per_worker, end)

<<<<<<< HEAD
            dataloader = DataLoader(dataset,
                                    places=place,
                                    num_workers=1,
                                    batch_size=1,
                                    drop_last=True,
                                    worker_init_fn=worker_spliter)
=======
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=1,
                batch_size=1,
                drop_last=True,
                worker_init_fn=worker_spliter,
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

            rets = []
            for d in dataloader:
                rets.append(d.numpy()[0][0])

            assert tuple(sorted(rets)) == tuple(range(0, 10))


if __name__ == '__main__':
    unittest.main()

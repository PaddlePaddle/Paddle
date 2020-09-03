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

from paddle.incubate.hapi.distributed import DistributedBatchSampler


class FakeDataset():
    def __init__(self):
        pass

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 10


class TestDistributedBatchSampler(unittest.TestCase):
    def test_sampler(self):
        dataset = FakeDataset()
        sampler = DistributedBatchSampler(dataset, batch_size=1, shuffle=True)
        for batch_idx in sampler:
            batch_idx
            pass

    def test_multiple_gpus_sampler(self):
        dataset = FakeDataset()
        sampler1 = DistributedBatchSampler(
            dataset, batch_size=4, shuffle=True, drop_last=True)
        sampler2 = DistributedBatchSampler(
            dataset, batch_size=4, shuffle=True, drop_last=True)

        sampler1.nranks = 2
        sampler1.local_rank = 0
        sampler1.num_samples = int(
            math.ceil(len(dataset) * 1.0 / sampler1.nranks))
        sampler1.total_size = sampler1.num_samples * sampler1.nranks

        sampler2.nranks = 2
        sampler2.local_rank = 1
        sampler2.num_samples = int(
            math.ceil(len(dataset) * 1.0 / sampler2.nranks))
        sampler2.total_size = sampler2.num_samples * sampler2.nranks

        for batch_idx in sampler1:
            batch_idx
            pass

        for batch_idx in sampler2:
            batch_idx
            pass


if __name__ == '__main__':
    unittest.main()

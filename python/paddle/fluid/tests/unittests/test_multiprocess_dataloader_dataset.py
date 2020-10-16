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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.io import TensorDataset, DataLoader
from paddle.fluid.dygraph.base import to_variable


class TestTensorDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            input_np = np.random.random([16, 3, 4]).astype('float32')
            input = to_variable(input_np)
            label_np = np.random.random([16, 1]).astype('int32')
            label = to_variable(label_np)

            dataset = TensorDataset([input, label])
            assert len(dataset) == 16
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True)

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, 3, 4]
                assert label.shape == [1, 1]
                assert isinstance(input, paddle.Tensor)
                assert isinstance(label, paddle.Tensor)
                assert np.allclose(input.numpy(), input_np[i])
                assert np.allclose(label.numpy(), label_np[i])

    def test_main(self):
        for p in [fluid.CPUPlace(), fluid.CUDAPlace(0)]:
            for num_workers in [0, 2]:
                ret = self.run_main(num_workers=num_workers, places=p)


if __name__ == '__main__':
    unittest.main()

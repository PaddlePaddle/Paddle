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

import paddle.fluid as fluid
import unittest
from paddle.fluid.dygraph import to_variable, Embedding, guard
import numpy as np


class TestImperativeUsingNonZeroGpu(unittest.TestCase):
    def run_main(self, np_arr, place):
        with guard(place):
            embedding = Embedding(size=[10, 10])
            var = to_variable(np_arr)
            self.assertTrue(np.array_equal(np_arr, var.numpy()))

    def test_non_zero_gpu(self):
        if not fluid.is_compiled_with_cuda():
            return

        np_arr = np.random.random([11, 13]).astype('float32')
        self.run_main(np_arr, fluid.CUDAPlace(1))
        self.run_main(np_arr, fluid.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()

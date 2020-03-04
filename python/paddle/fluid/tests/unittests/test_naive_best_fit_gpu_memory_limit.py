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
import numpy as np

fluid.core.globals()['FLAGS_allocator_strategy'] = 'naive_best_fit'

if fluid.is_compiled_with_cuda():
    fluid.core.globals()['FLAGS_gpu_memory_limit_mb'] = 10


class TestBase(unittest.TestCase):
    def setUp(self):
        if fluid.is_compiled_with_cuda():
            self._limit = fluid.core.globals()['FLAGS_gpu_memory_limit_mb']

    def test_allocate(self):
        if not fluid.is_compiled_with_cuda():
            return

        other_dim = int(1024 * 1024 / 4)

        place = fluid.CUDAPlace(0)
        t = fluid.LoDTensor()
        t.set(np.ndarray(
            [int(self._limit / 2), other_dim], dtype='float32'),
              place)
        del t

        t = fluid.LoDTensor()
        large_np = np.ndarray([2 * self._limit, other_dim], dtype='float32')

        try:
            t.set(large_np, place)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

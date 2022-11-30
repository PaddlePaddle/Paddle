# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
from paddle.fluid.memory_analysis import (
    pre_allocate_memory,
    get_max_memory_info,
)
from simple_nets import simple_fc_net


class TestMemoryAnalysis(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_get_memory_info(self):
        loss = simple_fc_net()
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
        optimizer.minimize(loss)
        main_prog = paddle.static.default_main_program()
        max_tmp_mem_1, max_persitable_mem_1 = get_max_memory_info(
            main_prog, batch_size=32
        )
        self.assertGreater(max_tmp_mem_1, 0)
        self.assertGreater(max_persitable_mem_1, 0)
        max_tmp_mem_2, max_persitable_mem_2 = get_max_memory_info(
            main_prog, batch_size=64
        )
        self.assertEqual(max_persitable_mem_1, max_persitable_mem_2)
        self.assertLess(max_tmp_mem_1, max_tmp_mem_2)


class TestPreAllocateMemory(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_pre_allocate(self):
        size = 32 * 1024 * 1024
        pre_allocate_memory(size, paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            pre_allocate_memory(size, paddle.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
import paddle.fluid.core as core
import numpy


class TestLoDArrayLength(unittest.TestCase):
    def test_array_length(self):
        tmp = layers.zeros(shape=[10], dtype='int32')
        i = layers.fill_constant(shape=[1], dtype='int64', value=10)
        arr = layers.array_write(tmp, i=i)
        arr_len = layers.array_length(arr)
        cpu = core.CPUPlace()
        exe = Executor(cpu)
        result = exe.run(fetch_list=[arr_len])[0]
        self.assertEqual(11, result[0])


if __name__ == '__main__':
    unittest.main()

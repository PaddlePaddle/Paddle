#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base
from paddle.pir_utils import test_with_pir_api


class TestIncrement(unittest.TestCase):
    @test_with_pir_api
    def test_api(self):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=5
            )
            expected_result = np.array([8], dtype='int64')

            output = paddle.tensor.math.increment(input, value=3)
            exe = base.Executor(base.CPUPlace())
            result = exe.run(fetch_list=[output])
            self.assertEqual((result == expected_result).all(), True)

        with base.dygraph.guard():
            input = paddle.ones(shape=[1], dtype='int64')
            expected_result = np.array([2], dtype='int64')
            output = paddle.tensor.math.increment(input, value=1)
            self.assertEqual((output.numpy() == expected_result).all(), True)


class TestInplaceApiWithDataTransform(unittest.TestCase):
    @test_with_pir_api
    def test_increment(self):
        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            with paddle.base.device_guard("gpu:0"):
                x = paddle.tensor.fill_constant([1], "float32", 0)
            with paddle.base.device_guard("cpu"):
                x = paddle.increment(x)
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            (a,) = exe.run(paddle.static.default_main_program(), fetch_list=[x])
            paddle.disable_static()
            self.assertEqual(a[0], 1)


if __name__ == "__main__":
    unittest.main()

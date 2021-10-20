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

from __future__ import print_function

import unittest
import numpy as np
import paddle
from paddle.static import program_guard, Program


# Test python API
class TestRandintLikeAPI(unittest.TestCase):
    def setUp(self):
        self.test_dtype = [
            "bool", "int8", "int16", "int32", "int64", "float16", "float32",
            "float64"
        ]
        self.x_inputs = []
        for dtype in self.test_dtype:
            self.x_inputs.append(np.zeros((10, 12)).astype(dtype))
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # results are from [-100, 100).
            test_x = []
            for dtype in self.test_dtype:
                x = paddle.fluid.data(
                    name="x_" + str(dtype), shape=[10, 12], dtype=dtype)
                test_x.append(x)

            test_outputs = []
            # x dtype ["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"]
            for x in test_x:
                x_output = []
                # self.test_dtype ["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"]
                for dtype in self.test_dtype:
                    out = paddle.randint_like(
                        x, low=-100, high=100, dtype=dtype)
                    x_output.append(out)
                test_outputs.append(x_output)

            exe = paddle.static.Executor(self.place)
            for x, output in zip(self.x_inputs, test_outputs):
                outs = exe.run(feed={'X': x}, fetch_list=output)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        # x dtype ["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        for x in self.x_inputs:
            x = paddle.to_tensor(x)
            # self.test_dtype ["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"]
            for dtype in self.test_dtype:
                out = paddle.randint_like(x, low=-100, high=100, dtype=dtype)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            test_x = []
            for dtype in self.test_dtype:
                x = paddle.fluid.data(
                    name="x_" + str(dtype), shape=[10, 12], dtype=dtype)
                test_x.append(x)

            # x dtype ["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"]  
            for x in test_x:
                # low is 5 and high is 5, low must less then high
                self.assertRaises(
                    ValueError, paddle.randint_like, x, low=5, high=5)
                # low(default value) is 0 and high is -5, low must less then high
                self.assertRaises(ValueError, paddle.randint_like, x, high=-5)
                # if high is None, low must be greater than 0
                self.assertRaises(ValueError, paddle.randint_like, x, low=-5)


if __name__ == "__main__":
    unittest.main()

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
from op_test import OpTest
import paddle
from paddle.fluid import core
from paddle.static import program_guard, Program


def output_hist(out):
    hist, _ = np.histogram(out, range=(-10, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestRandintLikeOp(OpTest):
    def setUp(self):
        self.op_type = "randint_like"
        x = np.zeros((10000, 784)).astype("float32")
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


# Test python API
class TestRandintLikeAPI(unittest.TestCase):
    def setUp(self):
        self.x_int32 = np.zeros((10, 12)).astype('int32')
        self.x_float32 = np.zeros((10, 12)).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # results are from [-100, 100).
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[10, 12], dtype='int32')
            x_float32 = paddle.fluid.data(
                name='x_float32', shape=[10, 12], dtype='float32')
            # x dtype is int32 and output dtype is 'int32'
            out1 = paddle.randint_like(
                x_int32, low=-100, high=100, dtype='int32')
            # x dtype is int32 and output dtype is 'int64'
            out2 = paddle.randint_like(
                x_int32, low=-100, high=100, dtype='int64')
            # x dtype is float32 and output dtype is 'int32'
            out3 = paddle.randint_like(
                x_float32, low=-100, high=100, dtype='int32')
            # x dtype is float32 and output dtype is 'int64'
            out4 = paddle.randint_like(
                x_float32, low=-100, high=100, dtype='int64')

            exe = paddle.static.Executor(self.place)
            outs_int32 = exe.run(feed={'X': self.x_int32},
                                 fetch_list=[out1, out2])
            outs_float32 = exe.run(feed={'X': self.x_float32},
                                   fetch_list=[out3, out4])

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x_int32 = paddle.to_tensor(self.x_int32)
        x_float32 = paddle.to_tensor(self.x_float32)
        # x dtype is int32 and output dtype is 'int32'
        out1 = paddle.randint_like(x_int32, low=-100, high=100, dtype='int32')
        # x dtype is int32 and output dtype is 'int64'
        out2 = paddle.randint_like(x_int32, low=-100, high=100, dtype='int64')
        # x dtype is float32 and output dtype is 'int32'
        out3 = paddle.randint_like(x_float32, low=-100, high=100, dtype='int32')
        # x dtype is float32 and output dtype is 'int64'
        out4 = paddle.randint_like(x_float32, low=-100, high=100, dtype='int64')
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(
                TypeError,
                paddle.randint_like,
                x_int32,
                high=5,
                dtype='float32')
            self.assertRaises(
                ValueError, paddle.randint_like, x_int32, low=5, high=5)
            self.assertRaises(ValueError, paddle.randint_like, x_int32, high=-5)
            self.assertRaises(ValueError, paddle.randint_like, x_int32, low=-5)


if __name__ == "__main__":
    unittest.main()

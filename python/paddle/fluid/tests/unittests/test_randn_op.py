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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestRandnOp(unittest.TestCase):
    def test_api(self):
        x1 = paddle.randn(shape=[1000, 784], dtype='float32')
        x2 = paddle.randn(shape=[1000, 784], dtype='float64')
        x3 = fluid.layers.fill_constant(
            shape=[1000, 784], dtype='float32', value=0)
        paddle.randn(shape=[1000, 784], out=x3, dtype='float32')
        x4 = paddle.randn(shape=[1000, 784], dtype='float32', device='cpu')
        x5 = paddle.randn(shape=[1000, 784], dtype='float32', device='gpu')
        x6 = paddle.randn(
            shape=[1000, 784],
            dtype='float32',
            device='gpu',
            stop_gradient=False)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(fluid.default_main_program(),
                      feed={},
                      fetch_list=[x1, x2, x3, x4, x5, x6])

        self.assertAlmostEqual(np.mean(res[0]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[0]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[1]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[1]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[2]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[2]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[3]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[3]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[4]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[4]), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res[5]), .0, delta=0.1)
        self.assertAlmostEqual(np.std(res[5]), 1., delta=0.1)


class TestRandnOpError(unittest.TestCase):
    def test_error(self):
        with program_guard(Program(), Program()):

            # The argument shape's size of randn_op should not be 0.
            def test_shape_size():
                out = paddle.randn(shape=[])

            self.assertRaises(AssertionError, test_shape_size)

            # The argument shape's type of randn_op should be list or tuple.
            def test_shape_type():
                out = paddle.randn(shape=1)

            self.assertRaises(TypeError, test_shape_type)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_float16():
                out = paddle.randn(shape=[1, 2], dtype='float16')

            self.assertRaises(TypeError, test_dtype_float16)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_int32():
                out = paddle.randn(shape=[1, 2], dtype='int32')

            self.assertRaises(TypeError, test_dtype_int32)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_int64():
                out = paddle.randn(shape=[1, 2], dtype='int64')

            self.assertRaises(TypeError, test_dtype_int64)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_uint8():
                out = paddle.randn(shape=[1, 2], dtype='uint8')

            self.assertRaises(TypeError, test_dtype_uint8)

            # The argument dtype of randn_op should be float32 or float64.
            def test_dtype_bool():
                out = paddle.randn(shape=[1, 2], dtype='bool')

            self.assertRaises(TypeError, test_dtype_bool)


if __name__ == "__main__":
    unittest.main()

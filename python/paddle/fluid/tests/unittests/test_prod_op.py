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

import paddle
import paddle.fluid as fluid
import unittest
import numpy as np


class TestProdOp(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        input = paddle.to_tensor(self.input)
        dy_result = paddle.prod(input)
        expected_result = np.prod(self.input)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

        dy_result = paddle.prod(input, axis=1)
        expected_result = np.prod(self.input, axis=1)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

        dy_result = paddle.prod(input, axis=[0, 1])
        expected_result = np.prod(self.input, axis=(0, 1))
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

        dy_result = paddle.prod(input, axis=1, keepdim=True)
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

        dy_result = paddle.prod(input, axis=1, dtype='int64')
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

        dy_result = paddle.prod(input, axis=1, keepdim=True, dtype='int64')
        expected_result = np.prod(
            self.input, axis=1, keepdims=True, dtype=np.int64)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

    def run_static(self, use_gpu=False):
        input = fluid.data(name='input', shape=[10, 10, 5], dtype='float32')
        result0 = paddle.prod(input)
        result1 = paddle.prod(input, axis=1)
        result2 = paddle.prod(input, axis=[0, 1])
        result3 = paddle.prod(input, axis=1, keepdim=True)
        result4 = paddle.prod(input, axis=1, dtype='int64')
        result5 = paddle.prod(input, axis=1, keepdim=True, dtype='int64')

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        static_result = exe.run(
            feed={"input": self.input},
            fetch_list=[result0, result1, result2, result3, result4, result5])

        expected_result = np.prod(self.input)
        self.assertTrue(np.allclose(static_result[0], expected_result))
        expected_result = np.prod(self.input, axis=1)
        self.assertTrue(np.allclose(static_result[1], expected_result))
        expected_result = np.prod(self.input, axis=(0, 1))
        self.assertTrue(np.allclose(static_result[2], expected_result))
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        self.assertTrue(np.allclose(static_result[3], expected_result))
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        self.assertTrue(np.allclose(static_result[4], expected_result))
        expected_result = np.prod(
            self.input, axis=1, keepdims=True, dtype=np.int64)
        self.assertTrue(np.allclose(static_result[5], expected_result))

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(6))
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)


class TestProdOpError(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = paddle.data(name='x', shape=[2, 2, 4], dtype='float32')
            bool_x = paddle.data(name='bool_x', shape=[2, 2, 4], dtype='bool')
            # The argument x shoule be a Tensor
            #self.assertRaises(AssertionError, paddle.prod, [1])

            # The argument axis's type shoule be int ,list or tuple
            self.assertRaises(TypeError, paddle.prod, x, 1.5)

            # The argument dtype of prod_op should be float32, float64, int32 or int64.
            self.assertRaises(TypeError, paddle.randn, x, 'int32')


if __name__ == "__main__":
    unittest.main()

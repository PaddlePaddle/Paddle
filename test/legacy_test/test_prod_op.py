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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from test_sum_op import TestReduceOPTensorAxisBase
from utils import dygraph_guard, static_guard

import paddle
from paddle.framework import core


class TestProdOp(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self, place):
        input = paddle.to_tensor(self.input, place=place)
        out = paddle.prod(input)
        expected_result = np.prod(self.input)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=1)
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=-1)
        expected_result = np.prod(self.input, axis=-1)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=[0, 1])
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(
            out.numpy(), expected_result, rtol=1e-05, atol=1e-8
        )

        out = paddle.prod(input, axis=1, keepdim=True)
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=1, dtype='int64')
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=1, keepdim=True, dtype='int64')
        expected_result = np.prod(
            self.input, axis=1, keepdims=True, dtype=np.int64
        )
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name='input', shape=[10, 10, 5], dtype='float32'
            )
            result0 = paddle.prod(input)
            result1 = paddle.prod(input, axis=1)
            result2 = paddle.prod(input, axis=-1)
            result3 = paddle.prod(input, axis=[0, 1])
            result4 = paddle.prod(input, axis=1, keepdim=True)
            result5 = paddle.prod(input, axis=1, dtype='int64')
            result6 = paddle.prod(input, axis=1, keepdim=True, dtype='int64')

            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            static_result = exe.run(
                feed={"input": self.input},
                fetch_list=[
                    result0,
                    result1,
                    result2,
                    result3,
                    result4,
                    result5,
                    result6,
                ],
            )

        expected_result = np.prod(self.input)
        np.testing.assert_allclose(
            static_result[0], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(
            static_result[1], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=-1)
        np.testing.assert_allclose(
            static_result[2], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(
            static_result[3], expected_result, rtol=1e-05, atol=1e-8
        )
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(
            static_result[4], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=1, dtype=np.int64)
        np.testing.assert_allclose(
            static_result[5], expected_result, rtol=1e-05
        )
        expected_result = np.prod(
            self.input, axis=1, keepdims=True, dtype=np.int64
        )
        np.testing.assert_allclose(
            static_result[6], expected_result, rtol=1e-05
        )

    def test_cpu(self):
        with dygraph_guard():
            self.run_imperative(place=paddle.CPUPlace())
        with static_guard():
            self.run_static()

    def test_gpu(self):
        if not paddle.base.core.is_compiled_with_cuda():
            return
        with dygraph_guard():
            self.run_imperative(place=paddle.CUDAPlace(0))
        with static_guard():
            self.run_static()


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestProdComplexOp(TestProdOp):
    def setUp(self):
        real = np.random.random(size=(10, 10, 5)).astype(np.float32)
        imag = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.input = real + 1j * imag

    def run_imperative(self, place):
        input = paddle.to_tensor(self.input, place=place)
        out = paddle.prod(input)
        expected_result = np.prod(self.input)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=1)
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

        out = paddle.prod(input, axis=[0, 1])
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(
            out.numpy(), expected_result, rtol=1e-05, atol=1e-8
        )

        out = paddle.prod(input, axis=1, keepdim=True)
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected_result, rtol=1e-05)

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name='input', shape=[10, 10, 5], dtype='complex64'
            )
            result0 = paddle.prod(input)
            result1 = paddle.prod(input, axis=1)
            result2 = paddle.prod(input, axis=-1)
            result3 = paddle.prod(input, axis=[0, 1])
            result4 = paddle.prod(input, axis=1, keepdim=True)

            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            static_complex_result = exe.run(
                feed={"input": self.input},
                fetch_list=[
                    result0,
                    result1,
                    result2,
                    result3,
                    result4,
                ],
            )

        expected_result = np.prod(self.input)
        np.testing.assert_allclose(
            static_complex_result[0], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=1)
        np.testing.assert_allclose(
            static_complex_result[1], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=-1)
        np.testing.assert_allclose(
            static_complex_result[2], expected_result, rtol=1e-05
        )
        expected_result = np.prod(self.input, axis=(0, 1))
        np.testing.assert_allclose(
            static_complex_result[3], expected_result, rtol=1e-05, atol=1e-8
        )
        expected_result = np.prod(self.input, axis=1, keepdims=True)
        np.testing.assert_allclose(
            static_complex_result[4], expected_result, rtol=1e-05
        )

    def test_cpu(self):
        with dygraph_guard():
            self.run_imperative(place=paddle.CPUPlace())
        with static_guard():
            self.run_static()

    def test_gpu(self):
        if not paddle.base.core.is_compiled_with_cuda():
            return
        with dygraph_guard():
            self.run_imperative(place=paddle.CUDAPlace(0))
        with static_guard():
            self.run_static()


class TestProdOpError(unittest.TestCase):

    def test_error(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[2, 2, 4], dtype='float32'
                )
                bool_x = paddle.static.data(
                    name='bool_x', shape=[2, 2, 4], dtype='bool'
                )
                # The argument x should be a Tensor
                self.assertRaises(TypeError, paddle.prod, [1])

                # The data type of x should be float32, float64, int32, int64
                self.assertRaises(TypeError, paddle.prod, bool_x)

                # The argument axis's type should be int ,list or tuple
                self.assertRaises(TypeError, paddle.prod, x, 1.5)

                # The argument dtype of prod_op should be float32, float64, int32 or int64.
                self.assertRaises(TypeError, paddle.prod, x, 'bool')


class TestProdWithTensorAxis1(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.prod
        self.np_api = np.prod
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')


class TestProdWithTensorAxis2(TestReduceOPTensorAxisBase):
    def init_data(self):
        self.pd_api = paddle.prod
        self.np_api = np.prod
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64'),
        ]


if __name__ == "__main__":
    unittest.main()

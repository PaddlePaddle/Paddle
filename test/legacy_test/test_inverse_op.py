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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


class TestInverseOp(OpTest):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse

    def setUp(self):
        self.op_type = "inverse"
        self.config()

        np.random.seed(123)
        mat = np.random.random(self.matrix_shape).astype(self.dtype)
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            mat = (
                np.random.random(self.matrix_shape)
                + 1j * np.random.random(self.matrix_shape)
            ).astype(self.dtype)

        inverse = np.linalg.inv(mat)

        self.inputs = {'Input': mat}
        self.outputs = {'Output': inverse}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseOpBatched(TestInverseOp):
    def config(self):
        self.matrix_shape = [8, 4, 4]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpLarge(TestInverseOp):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        self.check_grad(
            ['Input'], 'Output', max_relative_error=1e-6, check_pir=True
        )


class TestInverseOpFP32(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        self.check_grad(
            ['Input'], 'Output', max_relative_error=1e-2, check_pir=True
        )


class TestInverseOpBatchedFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [8, 4, 4]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpLargeFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpComplex64(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "complex64"
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseOpComplex128(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "complex128"
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float64"
            )
            result = paddle.inverse(x=input)
            input_np = np.random.random([4, 4]).astype("float64")
            result_np = np.linalg.inv(input_np)

            exe = base.Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(
                fetches[0], np.linalg.inv(input_np), rtol=1e-05
            )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 4]).astype("float64")
                input = paddle.to_tensor(input_np)
                result = paddle.inverse(input)
                np.testing.assert_allclose(
                    result.numpy(), np.linalg.inv(input_np), rtol=1e-05
                )


class TestInverseAPIError(unittest.TestCase):
    def test_errors(self):
        input_np = np.random.random([4, 4]).astype("float64")

        # input must be Variable.
        self.assertRaises(TypeError, paddle.inverse, input_np)

        # The data type of input must be float32 or float64.
        for dtype in ["bool", "int32", "int64", "float16"]:
            input = paddle.static.data(
                name='input_' + dtype, shape=[4, 4], dtype=dtype
            )
            self.assertRaises(TypeError, paddle.inverse, input)

        # The number of dimensions of input must be >= 2.
        input = paddle.static.data(name='input_2', shape=[4], dtype="float32")
        self.assertRaises(ValueError, paddle.inverse, input)


class TestInverseSingularAPI(unittest.TestCase):
    def setUp(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float64"
            )
            result = paddle.inverse(x=input)

            input_np = np.zeros([4, 4]).astype("float64")

            exe = base.Executor(place)
            try:
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )
            except RuntimeError as ex:
                print("The mat is singular")
            except ValueError as ex:
                print("The mat is singular")

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.ones([4, 4]).astype("float64")
                input = paddle.to_tensor(input_np)
                try:
                    result = paddle.inverse(input)
                except RuntimeError as ex:
                    print("The mat is singular")
                except ValueError as ex:
                    print("The mat is singular")


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

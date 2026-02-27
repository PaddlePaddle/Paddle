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

import unittest

import numpy as np
from op_test import OpTest, get_places

import paddle
from paddle import base


class TestInverseOp(OpTest):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float64"
        self.python_api = paddle.inverse

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
        self.python_api = paddle.inverse


class TestInverseOpZeroSize(TestInverseOp):
    def config(self):
        self.matrix_shape = [0, 0]
        self.dtype = "float64"
        self.python_api = paddle.inverse


class TestInverseOpBatchedZeroSize(TestInverseOp):
    def config(self):
        self.matrix_shape = [7, 0, 0]
        self.dtype = "float64"
        self.python_api = paddle.inverse


class TestInverseOpLarge(TestInverseOp):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float64"
        self.python_api = paddle.inverse

    def test_grad(self):
        self.check_grad(
            ['Input'], 'Output', max_relative_error=1e-6, check_pir=True
        )


class TestInverseOpFP32(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float32"
        self.python_api = paddle.inverse

    def test_grad(self):
        self.check_grad(
            ['Input'], 'Output', max_relative_error=1e-2, check_pir=True
        )


class TestInverseOpBatchedFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [8, 4, 4]
        self.dtype = "float32"
        self.python_api = paddle.inverse


class TestInverseOpLargeFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float32"
        self.python_api = paddle.inverse


class TestInverseOpComplex64(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "complex64"
        self.python_api = paddle.inverse

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseOpComplex128(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "complex128"
        self.python_api = paddle.inverse

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseOpBatchedComplex(TestInverseOp):
    def config(self):
        self.matrix_shape = [2, 3, 5, 5]
        self.dtype = "complex64"
        self.python_api = paddle.inverse

    def test_grad(self):
        self.check_grad(['Input'], 'Output', check_pir=True)


class TestInverseAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()

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

    def test_dygraph_with_name(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 4]).astype("float64")
                input = paddle.to_tensor(input_np)
                result = paddle.inverse(input, name='test_inverse')
                np.testing.assert_allclose(
                    result.numpy(), np.linalg.inv(input_np), rtol=1e-05
                )

    def test_static_with_name(self):
        for place in self.places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = paddle.static.data(
                    name="input", shape=[4, 4], dtype="float64"
                )
                result = paddle.inverse(x=input, name='test_inverse_static')
                input_np = np.random.random([4, 4]).astype("float64")
                exe = base.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )
                np.testing.assert_allclose(
                    fetches[0], np.linalg.inv(input_np), rtol=1e-05
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
        self.places = get_places()

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


class TestInverseAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 0]).astype("float64")
                input = paddle.to_tensor(input_np)
                input.stop_gradient = False
                result = paddle.linalg.inv(input)
                np_out = np.random.random([4, 0]).astype("float64")
                np.testing.assert_allclose(result.numpy(), np_out, rtol=1e-05)
                loss = paddle.sum(result)
                loss.backward()
                np.testing.assert_allclose(input.grad.shape, input.shape)


class TestInverseAPICompatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [6, 6]
        self.dtype = 'float64'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.random(self.shape).astype(self.dtype)
        # Ensure invertible
        while np.linalg.det(self.np_input) == 0:
            self.np_input = np.random.random(self.shape).astype(self.dtype)
        self.ref_output = np.linalg.inv(self.np_input)
        self.out_shape = self.np_input.shape

    def test_dygraph_compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []

        out1 = paddle.inverse(x)
        paddle_dygraph_out.append(out1)

        out2 = paddle.inverse(x=x)
        paddle_dygraph_out.append(out2)

        out3 = paddle.inverse(input=x)
        paddle_dygraph_out.append(out3)

        out4 = paddle.empty(self.out_shape)
        paddle.inverse(x, out=out4)
        paddle_dygraph_out.append(out4)

        out5 = x.inverse()
        paddle_dygraph_out.append(out5)

        ref_out = np.linalg.inv(self.np_input)
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_edge_cases(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.np_input)
        out = paddle.inverse(x)

        expected = np.linalg.inv(self.np_input)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_static_compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            out1 = paddle.inverse(x)
            out2 = paddle.inverse(x=x)
            out3 = paddle.inverse(input=x)

            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3],
            )
            ref_out = np.linalg.inv(self.np_input)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)

    def test_tensor_method_compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)

        out1 = x.inverse()
        out2 = x.inverse()
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_parameter_aliases(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)

        output_default = paddle.inverse(x)
        output_torch = paddle.inverse(input=x)

        np.testing.assert_allclose(
            output_default.numpy(), output_torch.numpy(), rtol=1e-5
        )

    def test_dimension_validation(self):
        paddle.disable_static()

        # 0D Tensor should raise ValueError
        scalar_input = paddle.to_tensor(1.0)
        with self.assertRaises(ValueError):
            paddle.inverse(scalar_input)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

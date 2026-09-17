#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from scipy import special

import paddle

np.random.seed(100)
paddle.seed(100)


def output_i0(x):
    return special.i0(x)


def ref_i0_grad(x, dout):
    gradx = special.i1(x)
    return dout * gradx


class TestI0API(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.out_ref = output_i0(self.x)
        self.place = get_places()

    def test_api_static(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.DTYPE
                )
                out = paddle.i0(x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[out],
                )
                np.testing.assert_allclose(res[0], self.out_ref, rtol=1e-5)
            paddle.disable_static()

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i0(x)

            out_ref = output_i0(self.x)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.i0, x)
            paddle.enable_static()


class TestI0Float32Zero2EightCase(TestI0API):
    DTYPE = "float32"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestI0Float32OverEightCase(TestI0API):
    DTYPE = "float32"
    DATA = [9, 10, 11, 12]


class TestI0Float64Zero2EightCase(TestI0API):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestI0Float64OverEightCase(TestI0API):
    DTYPE = "float64"
    DATA = [9, 10, 11, 12]


class TestI0Float64LargeInputCase(unittest.TestCase):
    DTYPE = "float64"
    X = np.array([713.0, 713.0], dtype=DTYPE)
    OUT_GRAD = np.array([1.0, 0.0], dtype=DTYPE)
    EXPECTED = np.array(
        [6.705128263670996e307, 6.705128263670996e307], dtype=DTYPE
    )
    EXPECTED_GRAD = np.array(
        [6.7004245591864025e307, 0.0], dtype=DTYPE
    )

    def setUp(self):
        self.place = get_places()

    def _check_result(self, out, x_grad):
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(x_grad).all())
        np.testing.assert_allclose(
            out, self.EXPECTED, rtol=1e-13, atol=0.0
        )
        np.testing.assert_allclose(
            x_grad, self.EXPECTED_GRAD, rtol=1e-13, atol=0.0
        )

    def test_api_static(self):
        for place in self.place:
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.X.shape, dtype=self.DTYPE
                )
                x.stop_gradient = False
                out_grad = paddle.static.data(
                    name="out_grad",
                    shape=self.OUT_GRAD.shape,
                    dtype=self.DTYPE,
                )
                out = paddle.i0(x)
                x_grad = paddle.static.gradients(out, x, out_grad)[0]
                exe = paddle.static.Executor(place)
                out_result, grad_result = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.X, "out_grad": self.OUT_GRAD},
                    fetch_list=[out, x_grad],
                )
            paddle.disable_static()
            self._check_result(out_result, grad_result)

    def test_api_dygraph(self):
        for place in self.place:
            paddle.disable_static(place)
            x = paddle.to_tensor(self.X, stop_gradient=False)
            out_grad = paddle.to_tensor(self.OUT_GRAD)
            out = paddle.i0(x)
            x_grad = paddle.grad(out, x, grad_outputs=out_grad)[0]
            out_result = out.numpy()
            grad_result = x_grad.numpy()
            paddle.enable_static()
            self._check_result(out_result, grad_result)


class TestI0Op(OpTest):
    def setUp(self) -> None:
        self.op_type = "i0"
        self.python_api = paddle.i0
        self.init_config()
        self.outputs = {"out": self.target}

    def init_config(self):
        self.dtype = np.float64
        zero_case = np.zeros(1).astype(self.dtype)
        rand_case = np.random.randn(100).astype(self.dtype)
        one2eight_case = np.random.uniform(low=1, high=8, size=100).astype(
            self.dtype
        )
        over_eight_case = np.random.uniform(low=9, high=15, size=100).astype(
            self.dtype
        )
        self.case = np.concatenate(
            [zero_case, rand_case, one2eight_case, over_eight_case]
        )
        self.inputs = {'x': self.case}
        self.target = output_i0(self.inputs['x'])

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[ref_i0_grad(self.case, 1 / self.case.size)],
            check_pir=True,
        )


class TestI0Op_ZeroSize(OpTest):
    def setUp(self) -> None:
        self.__class__.op_type = "i0"
        self.op_type = "i0"
        self.python_api = paddle.i0
        self.init_config()
        x = np.random.randn(3, 4, 0)
        self.inputs = {'x': x.astype(self.dtype)}
        self.attrs = {}
        self.outputs = {'out': special.i0(x)}

    def init_config(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x'], 'out')


class TestI0API_Compatibility(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.out = output_i0(self.x)
        self.place = get_places()

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.i0(x)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.i0(x=x)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.i0(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method kwargs
        out4 = x.i0()
        paddle_dygraph_out.append(out4)
        # Test out
        out5 = paddle.empty([])
        paddle.i0(x, out=out5)
        paddle_dygraph_out.append(out5)
        # scipy reference  out
        ref_out = output_i0(self.x)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)
        paddle.enable_static()

    def test_static_Compatibility(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.DTYPE
                )
                # Position args (args)
                out1 = paddle.i0(x)
                # Key words args (kwargs) for paddle
                out2 = paddle.i0(x=x)
                # Key words args for torch
                out3 = paddle.i0(input=x)
                # Tensor method args
                out4 = x.i0()

                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[out1, out2, out3, out4],
                )
                for out in fetches:
                    np.testing.assert_allclose(out, self.out, rtol=1e-5)
            paddle.disable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()

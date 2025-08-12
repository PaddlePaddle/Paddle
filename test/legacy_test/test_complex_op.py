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

import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import static
from paddle.base import core, dygraph

paddle.enable_static()


def ref_complex(x, y):
    return x + 1j * y


class TestComplexOp(OpTest):
    def init_spec(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "complex"
        self.python_api = paddle.complex
        self.init_spec()
        x = np.random.randn(*self.x_shape).astype(self.dtype)
        y = np.random.randn(*self.y_shape).astype(self.dtype)
        out_ref = ref_complex(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set('X'),
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_pir=True,
        )


class TestComplexOpBroadcast1(TestComplexOp):
    def init_spec(self):
        self.x_shape = [10, 3, 1, 4]
        self.y_shape = [100, 1]
        self.dtype = "float64"


class TestComplexOpBroadcast2(TestComplexOp):
    def init_spec(self):
        self.x_shape = [100, 1]
        self.y_shape = [10, 3, 1, 4]
        self.dtype = "float32"


class TestComplexOpBroadcast3(TestComplexOp):
    def init_spec(self):
        self.x_shape = [1, 100]
        self.y_shape = [100]
        self.dtype = "float32"


class TestComplexOpZeroSize1(TestComplexOp):
    def init_spec(self):
        self.x_shape = [1, 0]
        self.y_shape = [0]
        self.dtype = "float32"


class TestComplexOpZeroSize2(TestComplexOp):
    def init_spec(self):
        self.x_shape = [100, 1]
        self.y_shape = [10, 0, 1, 4]
        self.dtype = "float32"


class TestComplexOpZeroSize3(TestComplexOp):
    def init_spec(self):
        self.x_shape = [10, 3, 1, 0]
        self.y_shape = [100, 1]
        self.dtype = "float32"


class TestComplexOpZeroSize4(TestComplexOp):
    def init_spec(self):
        self.x_shape = [10, 3, 1, 0]
        self.y_shape = [0, 1]
        self.dtype = "float32"


class TestComplexAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(10, 10)
        self.y = np.random.randn(10, 10)
        self.out = ref_complex(self.x, self.y)

    def test_dygraph(self):
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            out_np = paddle.complex(x, y).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    def test_static(self):
        paddle.enable_static()
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[10, 10], dtype="float64")
            y = static.data("y", shape=[10, 10], dtype="float64")
            out = paddle.complex(x, y)

        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(
            mp, feed={"x": self.x, "y": self.y}, fetch_list=[out]
        )
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)


class OutTest(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_complex_api(self):
        def run_complex(test_type):
            x = paddle.arange(2, dtype=paddle.float32).unsqueeze(-1)
            y = paddle.arange(3, dtype=paddle.float32)
            x.stop_gradient = False
            y.stop_gradient = False
            z = paddle.ones([100])
            z.stop_gradient = False

            a = x + x
            b = y + y
            c = z + z

            if test_type == "return":
                c = paddle.complex(a, b)
            elif test_type == "input_out":
                paddle.complex(a, b, out=c)
            elif test_type == "both_return":
                c = paddle.complex(a, b, out=c)
            elif test_type == "both_input_out":
                tmp = paddle.complex(a, b, out=c)

            out = paddle._C_ops.complex(a, b)
            np.testing.assert_allclose(
                out.numpy(),
                c.numpy(),
                1e-20,
                1e-20,
            )

            d = c + c

            d.mean().backward()

            return c, x.grad, y.grad, z.grad

        paddle.disable_static()
        out1, x1, y1, z1 = run_complex("return")
        out2, x2, y2, z2 = run_complex("input_out")
        out3, x3, y3, z3 = run_complex("both_return")
        out4, x4, y4, z4 = run_complex("both_input_out")

        np.testing.assert_allclose(
            out1.numpy(),
            out2.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            out1.numpy(),
            out3.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            out1.numpy(),
            out4.numpy(),
            1e-20,
            1e-20,
        )

        np.testing.assert_allclose(
            x1.numpy(),
            x2.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            x1.numpy(),
            x3.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            x1.numpy(),
            x3.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            y1.numpy(),
            y2.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            y1.numpy(),
            y3.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            y1.numpy(),
            y4.numpy(),
            1e-20,
            1e-20,
        )
        np.testing.assert_equal(z1, None)
        np.testing.assert_equal(z2, None)
        np.testing.assert_equal(z3, None)
        np.testing.assert_equal(z4, None)


if __name__ == "__main__":
    unittest.main()

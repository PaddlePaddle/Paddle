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

<<<<<<< HEAD
import unittest

=======
from __future__ import print_function

import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
from op_test import OpTest

import paddle
<<<<<<< HEAD
from paddle import static
from paddle.fluid import dygraph
=======
from paddle.fluid import dygraph
from paddle import static
from paddle.fluid.framework import _test_eager_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


def ref_view_as_complex(x):
    real, imag = np.take(x, 0, axis=-1), np.take(x, 1, axis=-1)
    return real + 1j * imag


def ref_view_as_real(x):
    return np.stack([x.real, x.imag], -1)


class TestViewAsComplexOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "as_complex"
        self.python_api = paddle.as_complex
        x = np.random.randn(10, 10, 2).astype("float64")
        out_ref = ref_view_as_complex(x)
<<<<<<< HEAD
        self.out_grad = np.ones([10, 10], dtype="float64") + 1j * np.ones(
            [10, 10], dtype="float64"
        )
=======
        self.out_grad = np.ones(
            [10, 10], dtype="float64") + 1j * np.ones([10, 10], dtype="float64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[ref_view_as_real(self.out_grad)],
            user_defined_grad_outputs=[self.out_grad],
            check_eager=True,
        )


class TestViewAsRealOp(OpTest):
=======
        self.check_grad(['X'],
                        'Out',
                        user_defined_grads=[ref_view_as_real(self.out_grad)],
                        user_defined_grad_outputs=[self.out_grad],
                        check_eager=True)


class TestViewAsRealOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "as_real"
        real = np.random.randn(10, 10).astype("float64")
        imag = np.random.randn(10, 10).astype("float64")
        x = real + 1j * imag
        out_ref = ref_view_as_real(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}
        self.python_api = paddle.as_real
        self.out_grad = np.ones([10, 10, 2], dtype="float64")

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[ref_view_as_complex(self.out_grad)],
            user_defined_grad_outputs=[self.out_grad],
            check_eager=True,
        )


class TestViewAsComplexAPI(unittest.TestCase):
=======
        self.check_grad(['X'],
                        'Out',
                        user_defined_grads=[ref_view_as_complex(self.out_grad)],
                        user_defined_grad_outputs=[self.out_grad],
                        check_eager=True)


class TestViewAsComplexAPI(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.x = np.random.randn(10, 10, 2)
        self.out = ref_view_as_complex(self.x)

    def test_dygraph(self):
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.as_complex(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    def test_static(self):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[10, 10, 2], dtype="float64")
            out = paddle.as_complex(x)

        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={"x": self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

<<<<<<< HEAD

class TestViewAsRealAPI(unittest.TestCase):
=======
    def test_eager(self):
        with _test_eager_guard():
            self.test_dygraph()


class TestViewAsRealAPI(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        self.out = ref_view_as_real(self.x)

    def test_dygraph(self):
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.as_real(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    def test_static(self):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[10, 10], dtype="complex128")
            out = paddle.as_real(x)

        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={"x": self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

<<<<<<< HEAD
=======
    def test_eager(self):
        with _test_eager_guard():
            self.test_dygraph()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == "__main__":
    unittest.main()

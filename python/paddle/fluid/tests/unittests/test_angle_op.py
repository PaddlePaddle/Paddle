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
from paddle.fluid import dygraph
from paddle import static

paddle.enable_static()


def angle_grad(x, dout):
    if np.iscomplexobj(x):

        def angle_grad_element(xi, douti):
            if xi == 0:
                return 0
            rsquare = np.abs(xi)**2
            return -douti * xi.imag / rsquare + 1j * douti * xi.real / rsquare

        return np.vectorize(angle_grad_element)(x, dout)
    else:
        return np.zeros_like(x).astype(x.dtype)


class TestAngleOpFloat(OpTest):

    def setUp(self):
        self.op_type = "angle"
        self.python_api = paddle.angle
        self.dtype = "float64"
        self.x = np.linspace(-5, 5, 101).astype(self.dtype)
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'],
                        'Out',
                        user_defined_grads=[
                            angle_grad(self.x,
                                       np.ones_like(self.x) / self.x.size)
                        ],
                        check_eager=True)


class TestAngleOpComplex(OpTest):

    def setUp(self):
        self.op_type = "angle"
        self.python_api = paddle.angle
        self.dtype = "complex128"
        real = np.expand_dims(np.linspace(-2, 2, 11), -1).astype("float64")
        imag = np.linspace(-2, 2, 11).astype("float64")
        self.x = real + 1j * imag
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'],
                        'Out',
                        user_defined_grads=[
                            angle_grad(self.x,
                                       np.ones_like(self.x) / self.x.size)
                        ],
                        check_eager=True)


class TestAngleAPI(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
        self.out = np.angle(self.x)

    def test_dygraph(self):
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.angle(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    def test_static(self):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[2, 3], dtype="complex128")
            out = paddle.angle(x)

        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={"x": self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()

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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import static
from paddle.base import core, dygraph

paddle.enable_static()


def ref_view_as_complex(x):
    real, imag = np.take(x, 0, axis=-1), np.take(x, 1, axis=-1)
    return real + 1j * imag


def ref_view_as_real(x):
    return np.stack([x.real, x.imag], -1)


class TestViewAsComplexOp(OpTest):
    def setUp(self):
        self.op_type = "as_complex"
        self.python_api = paddle.as_complex
        x = np.random.randn(10, 10, 2).astype("float64")
        out_ref = ref_view_as_complex(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
        )


class TestViewAsRealOp(OpTest):
    def setUp(self):
        self.op_type = "as_real"
        real = np.random.randn(10, 10).astype("float64")
        imag = np.random.randn(10, 10).astype("float64")
        x = real + 1j * imag
        out_ref = ref_view_as_real(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}
        self.python_api = paddle.as_real

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
        )


class TestViewAsComplexAPI(unittest.TestCase):
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


class TestViewAsRealAPI(unittest.TestCase):
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


class TestViewAsRealAPI_ZeroSize(unittest.TestCase):
    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def setUp(self):
        self.x = np.random.randn(10, 0) + 1j * np.random.randn(10, 0)
        self.out = ref_view_as_real(self.x)

    def test_dygraph(self):
        for place in self.get_places():
            with dygraph.guard(place):
                x_tensor = paddle.to_tensor(self.x)
                x_tensor.stop_gradient = False
                out = paddle.as_real(x_tensor)
                np.testing.assert_allclose(self.out, out.numpy(), rtol=1e-05)
                out.sum().backward()
                np.testing.assert_allclose(x_tensor.grad.shape, x_tensor.shape)


if __name__ == "__main__":
    unittest.main()

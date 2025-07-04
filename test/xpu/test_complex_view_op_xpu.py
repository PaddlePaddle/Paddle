# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import static
from paddle.base import dygraph

paddle.enable_static()


def ref_view_as_complex(x):
    real, imag = np.take(x, 0, axis=-1), np.take(x, 1, axis=-1)
    return real + 1j * imag


def ref_view_as_real(x):
    return np.stack([x.real, x.imag], -1)


class XPUTestAsComplexOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'as_complex'
        self.use_dynamic_create_class = False

    class TestViewAsComplexOp(XPUOpTest):
        def setUp(self):
            self.op_type = "as_complex"
            self.python_api = paddle.as_complex
            x = np.random.randn(10, 10, 2).astype("float32")
            out_ref = ref_view_as_complex(x)
            self.inputs = {'X': x}
            self.outputs = {'Out': out_ref}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                )


class XPUTestAsRealOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'as_real'
        self.use_dynamic_create_class = False

    class TestViewAsRealOp(XPUOpTest):
        def setUp(self):
            self.op_type = "as_real"
            real = np.random.randn(10, 10).astype("float32")
            imag = np.random.randn(10, 10).astype("float32")
            x = real + 1j * imag
            out_ref = ref_view_as_real(x)
            self.inputs = {'X': x}
            self.outputs = {'Out': out_ref}
            self.python_api = paddle.as_real

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                )


support_types = get_xpu_op_support_types('as_complex')
for stype in support_types:
    create_test_class(globals(), XPUTestAsComplexOp, stype)

support_types = get_xpu_op_support_types('as_real')
for stype in support_types:
    create_test_class(globals(), XPUTestAsRealOp, stype)


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


if __name__ == "__main__":
    unittest.main()

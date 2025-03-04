#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from paddle.framework import core


def pow_grad(x, y, dout):
    dx = dout * y * np.power(x, (y - 1))
    dy = dout * np.log(x) * np.power(x, y)
    return dx, dy


class TestPowOp(OpTest):
    def setUp(self):
        self.op_type = "pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "comp"
        self.outputs = None
        self.custom_setting()
        if not self.outputs:
            self.outputs = {
                'Out': np.power(self.inputs['X'], self.attrs["factor"])
            }
        self.places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(core.CUDAPlace(0))

    def custom_setting(self):
        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float64"),
        }
        self.attrs = {"factor": 2.0}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim_pir=True,
            check_pir=True,
        )


class TestPowOp_ZeroDim1(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': np.random.uniform(1, 2, []).astype("float64"),
        }
        self.attrs = {"factor": float(np.random.uniform(1, 2, []))}


class TestPowOp_big_shape_1(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': np.random.uniform(1, 2, [10, 10]).astype("float64"),
        }
        self.attrs = {"factor": float(np.random.uniform(0, 10, []))}


class TestPowOp_big_shape_2(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': np.random.uniform(1, 2, [4, 6, 8]).astype("float64"),
        }
        self.attrs = {"factor": float(np.random.uniform(0, 10, []))}


class TestPowOpInt(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': np.asarray([1, 3, 6]),
        }
        self.attrs = {"factor": 2.0}

    def test_check_grad(self):
        pass


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": 2.0}

    def test_check_output(self):
        for place in self.places:
            self.check_output_with_place(
                place, check_pir=True, check_symbol_infer=False
            )

    def test_check_grad(self):
        for place in self.places:
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                check_pir=True,
            )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_1(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": -3.4}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_2(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": -2}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_3(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": -0.1}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_4(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": 0}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_5(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": 0.7}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_6(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": 1}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex64_7(TestPowOp_Complex64):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex64),
        }
        self.attrs = {"factor": 5.4}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128(TestPowOp):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": 2.0}

    def test_check_output(self):
        for place in self.places:
            self.check_output_with_place(
                place, check_pir=True, check_symbol_infer=False
            )

    def test_check_grad(self):
        for place in self.places:
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                check_pir=True,
            )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_1(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": -3.4}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_2(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": -2}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_3(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": -0.1}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_4(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": 0}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_5(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": 0.7}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_6(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": 1}


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestPowOp_Complex128_7(TestPowOp_Complex128):
    def custom_setting(self):
        self.inputs = {
            'X': (
                np.random.uniform(0.1, 1, [1, 3, 6])
                + 1j * np.random.uniform(0.1, 1, [1, 3, 6])
            ).astype(np.complex128),
        }
        self.attrs = {"factor": 5.4}


if __name__ == '__main__':
    unittest.main()

#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

numpy_apis = {
    "real": np.real,
    "imag": np.imag,
}

paddle_apis = {
    "real": paddle.real,
    "imag": paddle.imag,
}

paddle.enable_static()


class TestRealOp(OpTest):
    def setUp(self):
        self.op_type = "real"
        self.python_api = paddle.real
        self.init_type()
        self.init_input_output()
        self.init_grad_input_output()

    def init_type(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((20, 5)).astype(self.dtype)
            + 1j * np.random.random((20, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((20, 5), self.dtype)
        self.grad_x = np.real(self.grad_out) + 1j * np.zeros(
            self.grad_out.shape
        )

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
                user_defined_grads=[self.grad_x],
                user_defined_grad_outputs=[self.grad_out],
            )


class TestRealOpZeroSize1(TestRealOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random(0).astype(self.dtype)
            + 1j * np.random.random(0).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((0), self.dtype)
        self.grad_x = np.real(self.grad_out) + 1j * np.zeros(
            self.grad_out.shape
        )


class TestRealOpZeroSize2(TestRealOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((0, 5)).astype(self.dtype)
            + 1j * np.random.random((0, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((0, 5), self.dtype)
        self.grad_x = np.real(self.grad_out) + 1j * np.zeros(
            self.grad_out.shape
        )


class TestRealOpZeroSize3(TestRealOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((3, 0, 5)).astype(self.dtype)
            + 1j * np.random.random((3, 0, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((3, 0, 5), self.dtype)
        self.grad_x = np.real(self.grad_out) + 1j * np.zeros(
            self.grad_out.shape
        )


class TestImagOp(OpTest):
    def setUp(self):
        self.op_type = "imag"
        self.python_api = paddle.imag
        self.init_type()
        self.init_input_output()
        self.init_grad_input_output()

    def init_type(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((20, 5)).astype(self.dtype)
            + 1j * np.random.random((20, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((20, 5), self.dtype)
        self.grad_x = np.zeros(self.grad_out.shape) + 1j * np.real(
            self.grad_out
        )

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
                user_defined_grads=[self.grad_x],
                user_defined_grad_outputs=[self.grad_out],
            )


class TestImagOpZeroSize1(TestImagOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random(0).astype(self.dtype)
            + 1j * np.random.random(0).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((0), self.dtype)
        self.grad_x = np.zeros(self.grad_out.shape) + 1j * np.real(
            self.grad_out
        )


class TestImagOpZeroSize2(TestImagOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((0, 5)).astype(self.dtype)
            + 1j * np.random.random((0, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((0, 5), self.dtype)
        self.grad_x = np.zeros(self.grad_out.shape) + 1j * np.real(
            self.grad_out
        )


class TestImagOpZeroSize3(TestImagOp):
    def init_input_output(self):
        self.inputs = {
            'X': np.random.random((3, 0, 5)).astype(self.dtype)
            + 1j * np.random.random((3, 0, 5)).astype(self.dtype)
        }
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        self.grad_out = np.ones((3, 0, 5), self.dtype)
        self.grad_x = np.zeros(self.grad_out.shape) + 1j * np.real(
            self.grad_out
        )


if __name__ == "__main__":
    unittest.main()

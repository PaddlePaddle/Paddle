#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.executor import Executor
from paddle.base.framework import Program, grad_var_name

np.random.seed(123)
paddle.enable_static()


class PyArgsort:
    def __init__(self, input_shape, axis, descending, dtype):
        self.x = np.random.random(input_shape).astype(dtype)
        self.label = np.random.random(input_shape).astype(dtype)
        if axis < 0:
            self.axis = axis + len(self.x.shape)
        else:
            self.axis = axis
        self.descending = descending

    def forward(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(self.x, kind='quicksort', axis=self.axis), self.axis
            )
            self.sorted_x = np.flip(
                np.sort(self.x, kind='quicksort', axis=self.axis), self.axis
            )
        else:
            self.indices = np.argsort(self.x, kind='quicksort', axis=self.axis)
            self.sorted_x = np.sort(self.x, kind='quicksort', axis=self.axis)
        self.loss = self.sorted_x * self.label
        self.loss = np.sum(self.loss)
        out = (
            np.array(self.indices, dtype=self.indices.dtype),
            np.array(self.sorted_x, dtype=self.sorted_x.dtype),
            np.array(self.loss, dtype=self.loss.dtype),
        )
        return out


def create_tensor(np_data, place):
    tensor = core.DenseTensor()
    tensor.set(np_data, place)
    return tensor


class TestArgsortOpCPU(unittest.TestCase):
    def setup_program(self):
        self.main_program = Program()
        self.startup_program = Program()
        self.init_place()

    def setUp(self):
        paddle.enable_static()
        self.init_axis()
        self.init_datatype()
        self.init_direction()
        self.init_inputshape()

        self.setup_program()
        self.feed_data_field = {"x", "label"}
        self.grad_data_field = {"x"}

        self.py_argsort = PyArgsort(
            self.input_shape, self.axis, self.descending, self.dtype
        )

        with base.program_guard(self.main_program, self.startup_program):
            x = paddle.static.data(
                name="x", shape=[-1, *self.input_shape], dtype=self.dtype
            )
            x.stop_gradient = False
            x.desc.set_need_check_feed(False)
            label = paddle.static.data(
                name="label",
                shape=[-1, *list(self.input_shape)],
                dtype=self.dtype,
            )
            label.desc.set_need_check_feed(False)
            self.index = paddle.argsort(
                x=x, axis=self.axis, descending=self.descending
            )
            self.sorted_x = paddle.sort(
                x=x, axis=self.axis, descending=self.descending
            )
            self.sorted_x.stop_gradient = False
            loss = paddle.multiply(self.sorted_x, label)
            self.loss = paddle.sum(loss)

    def forward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_argsort, x), self.place)
            for x in self.feed_data_field
        }
        exe = Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=[self.index, self.sorted_x, self.loss],
        )
        return out

    def backward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_argsort, x), self.place)
            for x in self.feed_data_field
        }
        fetch_list = [
            self.main_program.global_block().var(grad_var_name(x))
            for x in self.grad_data_field
        ]
        exe = Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=fetch_list,
            return_numpy=False,
        )
        return out

    def test_backward(self, numeric_grad_delta=1e-5, max_relative_error=1e-7):
        self.check_forward()

        with base.program_guard(self.main_program, self.startup_program):
            append_backward(self.loss)

        ana_grad = [np.array(x) for x in self.backward()]

        num_grad = self.get_numerical_gradient(delta=numeric_grad_delta)
        self.assert_is_close(
            num_grad,
            ana_grad,
            'x',
            max_relative_error=max_relative_error,
            msg_prefix=f"Gradient Check On {self.place}",
        )

    def check_forward(self):
        pd_outputs = self.forward()
        py_outputs = self.py_argsort.forward()
        for pd_output, py_output in zip(pd_outputs, py_outputs):
            self.assertEqual(pd_output.shape, py_output.shape)
            np.testing.assert_allclose(
                pd_output, py_output, rtol=1e-05, atol=0, equal_nan=False
            )

    def get_numerical_gradient(self, delta=1e-7):
        if self.dtype == 'float16':
            delta = np.array(delta).astype(np.float16)
        feed_list = [getattr(self.py_argsort, x) for x in self.grad_data_field]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=['readwrite']):
                o = float(f)
                f[...] = o + delta
                y_pos = self.forward()[2]

                f[...] = o - delta
                y_neg = self.forward()[2]

                f[...] = o
                dout_dfeed = (y_pos - y_neg) / (delta * 2)
                g[...] = dout_dfeed

        return grad_list

    def assert_is_close(
        self,
        numeric_grads,
        analytic_grads,
        names,
        max_relative_error,
        msg_prefix,
    ):
        for a, b, name in zip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return (
                    f"argsort error, {msg_prefix} variable {name} max gradient diff {max_diff:f} over limit {max_relative_error:f}, "
                    f"the first error element is {a.flatten()[offset]}, expected {b.flatten()[offset]:f}, but got {a.flatten()[offset]:f}."
                )

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def init_axis(self):
        self.axis = -1

    def init_datatype(self):
        self.dtype = "float64"

    def init_direction(self):
        self.descending = False

    def init_inputshape(self):
        self.input_shape = (2, 2, 2, 2, 3)

    def init_place(self):
        self.place = core.CPUPlace()


class TestArgsortOpGPU(TestArgsortOpCPU):
    def init_place(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()


class TestArgsortOpAxis0CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis0GPU(TestArgsortOpGPU):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis1GPU(TestArgsortOpGPU):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxis2GPU(TestArgsortOpGPU):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg1GPU(TestArgsortOpGPU):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpAxisNeg2GPU(TestArgsortOpGPU):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisCPU(TestArgsortOpCPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisGPU(TestArgsortOpGPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0CPU(TestArgsortOpAxis0CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0GPU(TestArgsortOpAxis0GPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1CPU(TestArgsortOpAxis1CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1GPU(TestArgsortOpAxis1GPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2CPU(TestArgsortOpAxis2CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2GPU(TestArgsortOpAxis2GPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1CPU(TestArgsortOpAxisNeg1CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1GPU(TestArgsortOpAxisNeg1GPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2CPU(TestArgsortOpAxisNeg2CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2GPU(TestArgsortOpAxisNeg2GPU):
    def init_direction(self):
        self.descending = True


if __name__ == "__main__":
    unittest.main()

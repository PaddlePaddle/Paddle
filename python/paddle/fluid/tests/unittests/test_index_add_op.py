# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import paddle
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def compute_index_add_ref(axis, x_shape, x_np, add_value_shape, add_value_np,
                          index_size, index_np):
    if axis < 0:
        axis = axis + len(x_shape)
    if axis != 0:
        outer_loop = np.prod(x_shape[:axis]).astype(int)
        x_reshape = [outer_loop] + list(x_shape[axis:])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))

        add_value_reshape = [np.prod(add_value_shape[:axis]).astype(int)
                             ] + list(add_value_shape[axis:])

        add_value_np_reshape = np.reshape(add_value_np,
                                          tuple(add_value_reshape))
    else:
        x_np_reshape = x_np
        add_value_np_reshape = add_value_np
    out_np = x_np_reshape.copy()

    if axis != 0:
        for i in range(outer_loop):
            for j in range(index_size):
                out_np[i, index_np[j]] += add_value_np_reshape[i, j]
    else:
        for j in range(index_size):
            out_np[index_np[j]] += add_value_np_reshape[j]
    ref_out = np.reshape(out_np, x_shape)
    return ref_out


def raw_index_add(x, index, value, axis):
    return paddle.index_add(x, index, axis, value)


class TestIndexAddOp(OpTest):

    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.init_dtype_type()
        index_np = np.random.randint(low=0,
                                     high=self.x_shape[self.axis],
                                     size=self.index_size)
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type)

        self.inputs = {'X': x_np, 'Index': index_np, 'AddValue': add_value_np}
        self.attrs = {'axis': self.axis}
        out = compute_index_add_ref(self.axis, self.x_shape, x_np,
                                    self.add_value_shape, add_value_np,
                                    self.index_size, index_np)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (101, 3)
        self.index_size = 3
        self.add_value_shape = (3, 3)

    def test_check_output(self):
        self.check_output(check_eager=True, atol=1e-2)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'AddValue'], 'Out', check_eager=True)


class TestIndexAddAPI(unittest.TestCase):

    def setUp(self):
        self.setType()
        self.setPlace()
        self.config()
        self.check_backward = True
        self.generate_input_data()

        self.index_shape = tuple([self.index_size])

        self.rtol = 1e-5
        self.atol = 1e-2
        if self.x_type is np.float16:
            self.atol = 1e-1

    def setType(self):
        self.x_type = np.float32
        self.index_type = np.int32

    def setPlace(self):
        self.place = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def config(self):
        self.axis = 0
        self.x_shape = (100, 5)
        self.index_size = 20
        self.add_value_shape = (20, 5)

    def generate_input_data(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)

        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type)
        self.index_np = np.random.randint(low=0,
                                          high=self.x_shape[axis],
                                          size=self.index_size).astype(
                                              self.index_type)
        if self.check_backward:
            self.dout_np = np.random.random(self.x_shape).astype(self.x_type)

    def compute_index_add_backward_ref(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)

        x_grad = self.dout_np

        dout_tensor = paddle.to_tensor(self.dout_np)
        index = paddle.to_tensor(self.index_np)
        add_value_grad = paddle.index_select(dout_tensor, index, axis)

        return x_grad, add_value_grad.numpy()

    def run_imperative(self, device):
        paddle.device.set_device(device)
        input_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
        index = paddle.to_tensor(self.index_np)
        add_value = paddle.to_tensor(self.add_value_np, stop_gradient=False)

        out = paddle.index_add(input_tensor, index, self.axis, add_value)
        ref_out = compute_index_add_ref(self.axis, self.x_shape, self.x_np,
                                        self.add_value_shape, self.add_value_np,
                                        self.index_size, self.index_np)
        np.testing.assert_allclose(ref_out,
                                   out.numpy(),
                                   rtol=self.rtol,
                                   atol=self.atol)

        if self.check_backward:
            dout_tensor = paddle.to_tensor(self.dout_np)
            paddle.autograd.backward([out], [dout_tensor], retain_graph=True)
            ref_x_grad, ref_add_value_grad = self.compute_index_add_backward_ref(
            )
            np.testing.assert_allclose(ref_x_grad,
                                       input_tensor.grad.numpy(),
                                       rtol=self.rtol,
                                       atol=self.atol)
            np.testing.assert_allclose(ref_add_value_grad,
                                       add_value.grad.numpy(),
                                       rtol=self.rtol,
                                       atol=self.atol)

    def run_static(self, device):
        x = paddle.static.data(name='X', shape=self.x_shape, dtype=self.x_type)
        index = paddle.static.data(name='Index',
                                   shape=self.index_shape,
                                   dtype=self.index_type)
        add_value = paddle.static.data(name='AddValue',
                                       shape=self.add_value_shape,
                                       dtype=self.x_type)

        out = paddle.index_add(x, index, self.axis, add_value)

        if device == "cpu":
            place = paddle.CPUPlace()
        elif device == "gpu":
            place = paddle.CUDAPlace(0)
        else:
            raise TypeError(
                "paddle.index_add api only support cpu and gpu device now.")

        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        res = exe.run(paddle.static.default_main_program(),
                      feed={
                          "X": self.x_np,
                          "Index": self.index_np,
                          "AddValue": self.add_value_np,
                      },
                      fetch_list=[out.name],
                      return_numpy=False)
        return res

    def test_static(self):
        paddle.enable_static()
        for device in self.place:
            with paddle.static.program_guard(Program()):
                out = self.run_static(device)
            ref_out = compute_index_add_ref(self.axis, self.x_shape, self.x_np,
                                            self.add_value_shape,
                                            self.add_value_np, self.index_size,
                                            self.index_np)
            np.testing.assert_allclose(ref_out,
                                       np.array(out[0]),
                                       rtol=self.rtol,
                                       atol=self.atol)

    def test_dynamic(self):
        paddle.disable_static()
        for device in self.place:
            self.run_imperative(device)


class TestIndexAddAPIMoreType(TestIndexAddAPI):

    def setType(self):
        self.x_type = np.float64
        self.index_type = np.int64


class TestIndexAddAPICase2(TestIndexAddAPI):

    def config(self):
        self.axis = 1
        self.x_shape = (100, 100, 5)
        self.index_size = 20
        self.add_value_shape = (100, 20, 5)


class TestIndexAddAPICase3(TestIndexAddAPI):

    def config(self):
        self.axis = 2
        self.x_shape = (100, 100, 25)
        self.index_size = 20
        self.add_value_shape = (100, 100, 20)


class TestIndexAddAPICase4(TestIndexAddAPI):

    def config(self):
        self.axis = 0
        self.x_shape = (10, )
        self.index_size = 4
        self.add_value_shape = (4, )


class TestIndexAddAPICase5(TestIndexAddAPI):

    def config(self):
        self.axis = -1
        self.x_shape = (10, 10)
        self.index_size = 4
        self.add_value_shape = (10, 4)


class TestIndexAddAPIError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):

            def test_add_value_shape():
                axis = 0
                x = paddle.static.data(name='X',
                                       shape=[10, 10],
                                       dtype="float64")
                index = paddle.static.data(name='Index',
                                           shape=[4],
                                           dtype="int32")
                add_value = paddle.static.data(name='AddValue',
                                               shape=[4, 3],
                                               dtype="float64")
                out = paddle.index_add(x, index, axis, add_value)

            self.assertRaises(ValueError, test_add_value_shape)

            def test_index_dtype():
                axis = 0
                x = paddle.static.data(name='X1',
                                       shape=[10, 10],
                                       dtype="float64")
                index = paddle.static.data(name='Index1',
                                           shape=[4],
                                           dtype="float32")
                add_value = paddle.static.data(name='AddValue1',
                                               shape=[4, 10],
                                               dtype="float64")
                out = paddle.index_add(x, index, axis, add_value)

            self.assertRaises(TypeError, test_index_dtype)

            def test_index_shape():
                axis = 0
                x = paddle.static.data(name='X2',
                                       shape=[10, 10],
                                       dtype="float64")
                index = paddle.static.data(name='Index2',
                                           shape=[4, 3],
                                           dtype="int32")
                add_value = paddle.static.data(name='AddValue2',
                                               shape=[4, 10],
                                               dtype="float64")
                out = paddle.index_add(x, index, axis, add_value)

            self.assertRaises(ValueError, test_index_shape)

            def test_axis_value():
                axis = 3
                x = paddle.static.data(name='X3',
                                       shape=[10, 10],
                                       dtype="float64")
                index = paddle.static.data(name='Index3',
                                           shape=[4],
                                           dtype="int32")
                add_value = paddle.static.data(name='AddValue3',
                                               shape=[4, 10],
                                               dtype="float64")
                out = paddle.index_add(x, index, axis, add_value)

            self.assertRaises(ValueError, test_axis_value)

            def test_add_value_broadcast():
                axis = 0
                x = paddle.static.data(name='X4',
                                       shape=[10, 10],
                                       dtype="float64")
                index = paddle.static.data(name='Index4',
                                           shape=[4],
                                           dtype="int32")
                add_value = paddle.static.data(name='AddValue4',
                                               shape=[4],
                                               dtype="float64")
                out = paddle.index_add(x, index, axis, add_value)

            self.assertRaises(ValueError, test_add_value_broadcast)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


class TestUnStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        pass

    def get_y_names(self):
        y_names = []
        for i in range(self.input_dim[self.axis]):
            y_names.append(f'y{i}')
        return y_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'unstack'
        self.python_api = paddle.unstack
        self.x = np.random.random(size=self.input_dim).astype(self.dtype)

        outs = np.split(self.x, self.input_dim[self.axis], self.axis)
        new_shape = list(self.input_dim)
        del new_shape[self.axis]
        y_names = self.get_y_names()
        tmp = []
        tmp_names = []
        for i in range(self.input_dim[self.axis]):
            tmp.append((y_names[i], np.reshape(outs[i], new_shape)))
            tmp_names.append(y_names[i])

        self.python_out_sig = tmp_names
        self.inputs = {'X': self.x}
        self.outputs = {'Y': tmp}
        self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], self.get_y_names())


class TestUnStackFP16Op(TestUnStackOpBase):
    def initParameters(self):
        self.dtype = np.float16


class TestStackFP16Op3(TestUnStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = -1


class TestStackFP16Op4(TestUnStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = -3


class TestStackFP16Op5(TestUnStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = 1


class TestStackFP16Op6(TestUnStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = 2


class TestStackOp3(TestUnStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestUnStackOpBase):
    def initParameters(self):
        self.axis = -3


class TestStackOp5(TestUnStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestUnStackOpBase):
    def initParameters(self):
        self.axis = 2


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestUnStackBF16Op(OpTest):
    def initDefaultParameters(self):
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = np.uint16

    def initParameters(self):
        pass

    def get_y_names(self):
        y_names = []
        for i in range(self.input_dim[self.axis]):
            y_names.append(f'y{i}')
        return y_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'unstack'
        self.python_api = paddle.unstack
        self.x = np.random.random(size=self.input_dim).astype(np.float32)
        outs = np.split(self.x, self.input_dim[self.axis], self.axis)
        new_shape = list(self.input_dim)
        del new_shape[self.axis]
        y_names = self.get_y_names()
        tmp = []
        tmp_names = []
        for i in range(self.input_dim[self.axis]):
            tmp.append(
                (
                    y_names[i],
                    np.reshape(convert_float_to_uint16(outs[i]), new_shape),
                )
            )
            tmp_names.append(y_names[i])

        self.x = convert_float_to_uint16(self.x)
        self.python_out_sig = tmp_names
        self.inputs = {'X': self.x}
        self.outputs = {'Y': tmp}
        self.attrs = {'axis': self.axis, 'num': self.input_dim[self.axis]}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(self.inputs['X'])
            x.stop_gradient = False
            y = paddle.unstack(
                x, axis=self.attrs['axis'], num=self.attrs['num']
            )
            dx = paddle.grad(y, x)[0].numpy()
            dx_expected = convert_float_to_uint16(
                np.ones(self.input_dim, np.float32)
            )
            np.testing.assert_array_equal(dx, dx_expected)


class TestUnstackZeroInputOp(unittest.TestCase):
    def unstack_zero_input_static(self):
        paddle.enable_static()

        array = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
        paddle.unstack(x, axis=1)

    def unstack_zero_input_dynamic(self):
        array = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(array, [0]), dtype='float32')
        paddle.unstack(x, axis=1)

    def test_type_error(self):
        paddle.disable_static()

        self.assertRaises(ValueError, self.unstack_zero_input_dynamic)
        self.assertRaises(ValueError, self.unstack_zero_input_static)

        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()

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
from op_test import OpTest, convert_float_to_uint16
from utils import dygraph_guard, static_guard

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
        self.prim_op_type = "comp"
        self.python_api = paddle.unstack
        self.public_python_api = paddle.unstack
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
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], self.get_y_names(), check_pir=True, check_prim_pir=True
        )


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


class TestStackOp3_Complex64(TestStackOp3):
    def initParameters(self):
        self.dtype = np.complex64
        self.axis = -1


class TestStackOp4_complex64(TestStackOp4):
    def initParameters(self):
        self.dtype = np.complex64
        self.axis = -3


class TestStackOp5_complex64(TestStackOp5):
    def initParameters(self):
        self.dtype = np.complex64
        self.axis = 1


class TestStackOp6_complex64(TestStackOp6):
    def initParameters(self):
        self.dtype = np.complex64
        self.axis = 2


class TestStackOp3_Complex128(TestStackOp3):
    def initParameters(self):
        self.dtype = np.complex128
        self.axis = -1


class TestStackOp4_complex128(TestStackOp4):
    def initParameters(self):
        self.dtype = np.complex128
        self.axis = -3


class TestStackOp5_complex128(TestStackOp5):
    def initParameters(self):
        self.dtype = np.complex128
        self.axis = 1


class TestStackOp6_complex128(TestStackOp6):
    def initParameters(self):
        self.dtype = np.complex128
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
        self.prim_op_type = "comp"
        self.python_api = paddle.unstack
        self.public_python_api = paddle.unstack
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
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)

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

        dtypes = ['float32', 'complex64', 'complex128']
        for dtype in dtypes:
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                data = np.random.random([0]).astype(dtype)
                if dtype == 'complex64' or dtype == 'complex128':
                    data = (
                        np.random.random([0]) + 1j * np.random.random([0])
                    ).astype(dtype)
                x = paddle.static.data(shape=[0], dtype=dtype, name='x')
                paddle.unstack(x, axis=1)

    def unstack_zero_input_dynamic(self):
        paddle.disable_static()
        dtypes = ['float32', 'complex64', 'complex128']
        for dtype in dtypes:
            with base.dygraph.guard():
                data = np.random.random([0]).astype(dtype)
                if dtype == 'complex64' or dtype == 'complex128':
                    data = (
                        np.random.random([0]) + 1j * np.random.random([0])
                    ).astype(dtype)
                x = paddle.to_tensor(data)
                paddle.unstack(x, axis=1)

    def test_type_error(self):
        paddle.disable_static()

        self.assertRaises(ValueError, self.unstack_zero_input_dynamic)
        self.assertRaises(ValueError, self.unstack_zero_input_static)

        paddle.disable_static()


class TestUnstackEmptyTensorInput(unittest.TestCase):
    def _get_places(self):
        places = [paddle.base.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.base.CUDAPlace(0))
        return places

    def _generate_empty_tensor(self, shape):
        return np.empty(shape)

    def _test_unstack_with_shapes(self, shape, axis, place=None):
        empty_tensor = self._generate_empty_tensor(shape)

        # NOTE: Use `numpy.unstack` if you are using NumPy version 2.1.0 or later.
        # out_ref = np.unstack(empty_tensor, axis)
        out_ref = tuple(np.moveaxis(empty_tensor, axis, 0))

        if place is None:  # Dygraph mode
            tensor = paddle.to_tensor(empty_tensor)
            result = paddle.unstack(tensor, axis=axis)
        else:  # Static mode
            with paddle.static.program_guard(paddle.static.Program()):
                data_tensor = paddle.static.data(
                    shape=shape, dtype='float64', name='x'
                )
                result = paddle.unstack(data_tensor, axis=axis)
                exe = paddle.base.Executor(place=place)
                feed_dict = {'x': empty_tensor}
                result = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=result,
                )

        # Assert the number of unstacked tensors
        self.assertEqual(len(out_ref), len(result))
        # Assert the shape of each unstacked tensor
        for ref, res in zip(out_ref, result):
            np.testing.assert_array_equal(ref.shape, res.shape)

    def test_unstack_with_dygraph_empty_tensor_input(self):
        with dygraph_guard():
            self._test_unstack_with_shapes((0,), axis=0)
            self._test_unstack_with_shapes((5, 0), axis=1)
            self._test_unstack_with_shapes((5, 0, 10), axis=2)
            self._test_unstack_with_shapes((7, 11, 0), axis=1)
            self._test_unstack_with_shapes((0, 11, 22), axis=-2)

    def _test_unstack_with_static_empty_tensor_input(self, place):
        with static_guard():
            self._test_unstack_with_shapes((0,), axis=0, place=place)
            self._test_unstack_with_shapes((5, 0), axis=1, place=place)
            self._test_unstack_with_shapes((5, 0, 10), axis=2, place=place)
            self._test_unstack_with_shapes((7, 11, 0), axis=1, place=place)
            self._test_unstack_with_shapes((0, 11, 22), axis=-2, place=place)

    def test_unstack_with_static_empty_tensor_input(self):
        for place in self._get_places():
            self._test_unstack_with_static_empty_tensor_input(place)


if __name__ == '__main__':
    unittest.main()

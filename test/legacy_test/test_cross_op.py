# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import Program, core, program_guard


class TestCrossOp(OpTest):
    def setUp(self):
        self.op_type = "cross"
        self.python_api = paddle.cross
        self.initTestCase()
        self.inputs = {
            'X': np.random.random(self.shape).astype(self.dtype),
            'Y': np.random.random(self.shape).astype(self.dtype),
        }
        self.init_output()

    def initTestCase(self):
        self.attrs = {'dim': -2}
        self.dtype = np.float64
        self.shape = (1024, 3, 1)

    def init_output(self):
        x = np.squeeze(self.inputs['X'], 2)
        y = np.squeeze(self.inputs['Y'], 2)
        z_list = []
        for i in range(1024):
            z_list.append(np.cross(x[i], y[i]))
        self.outputs = {'Out': np.array(z_list).reshape(self.shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestCrossOpCase1(TestCrossOp):
    def initTestCase(self):
        self.shape = (2048, 3)
        self.dtype = np.float32

    def init_output(self):
        z_list = []
        for i in range(2048):
            z_list.append(np.cross(self.inputs['X'][i], self.inputs['Y'][i]))
        self.outputs = {'Out': np.array(z_list).reshape(self.shape)}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCrossFP16Op(TestCrossOp):
    def initTestCase(self):
        self.shape = (2048, 3)
        self.dtype = np.float16

    def init_output(self):
        z_list = []
        for i in range(2048):
            z_list.append(np.cross(self.inputs['X'][i], self.inputs['Y'][i]))
        self.outputs = {'Out': np.array(z_list).reshape(self.shape)}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestCrossBF16Op(OpTest):
    def setUp(self):
        self.op_type = "cross"
        self.python_api = paddle.cross
        self.initTestCase()
        self.x = np.random.random(self.shape).astype(np.float32)
        self.y = np.random.random(self.shape).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.init_output()

    def initTestCase(self):
        self.attrs = {'dim': -2}
        self.dtype = np.uint16
        self.shape = (1024, 3, 1)

    def init_output(self):
        x = np.squeeze(self.x, 2)
        y = np.squeeze(self.y, 2)
        z_list = []
        for i in range(1024):
            z_list.append(np.cross(x[i], y[i]))
        out = np.array(z_list).astype(np.float32).reshape(self.shape)
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')


class TestCrossAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        ).astype('float32')
        self.data_y = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ).astype('float32')

    def test_cross_api(self):
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[-1, 3], dtype="float32")
            z = paddle.cross(x, y, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'y': self.data_y},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[-1, 3], dtype="float32")
            z = paddle.cross(x, y)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'y': self.data_y},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.array(
            [[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]
        )
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 3:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')

            y_1 = paddle.cross(x, y, name='result')
            self.assertEqual(('result' in y_1.name), True)

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        # with base.dygraph.guard():
        #     x = base.dygraph.to_variable(self.data_x)
        #     y = base.dygraph.to_variable(self.data_y)
        #     z = paddle.cross(x, y)
        #     np_z = z.numpy()
        # expect_out = np.array([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0],
        #                        [-1.0, -1.0, -1.0]])
        # np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with base.dygraph.guard():
            x = base.dygraph.to_variable(self.data_x)
            y = base.dygraph.to_variable(self.data_y)
            z = paddle.cross(x, y, axis=1)
            np_z = z.numpy()
        expect_out = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

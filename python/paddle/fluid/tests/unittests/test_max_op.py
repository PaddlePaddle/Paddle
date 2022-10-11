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

import os
import unittest
import tempfile
import numpy as np
from op_test import OpTest, skip_check_grad_ci, check_out_dtype
import paddle
from paddle.fluid.framework import _test_eager_guard
import paddle.fluid.core as core
import paddle.inference as paddle_infer
from test_sum_op import TestReduceOPTensorAxisBase


class ApiMaxTest(unittest.TestCase):

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="float32")
            result_max = paddle.max(x=data, axis=1)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.rand(10, 10).astype(np.float32)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=1)).all(), True)

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_max = paddle.max(x=data, axis=0)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=0)).all(), True)

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_max = paddle.max(x=data, axis=(0, 1))
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_max])
        self.assertEqual((res == np.max(input_data, axis=(0, 1))).all(), True)

    def test_errors(self):
        paddle.enable_static()

        def test_input_type():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                data = np.random.rand(10, 10)
                result_max = paddle.max(x=data, axis=0)

        self.assertRaises(TypeError, test_input_type)

    def test_imperative_api(self):
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_tensor(np_x)
        z = paddle.max(x, axis=0)
        np_z = z.numpy()
        z_expected = np.array(np.max(np_x, axis=0))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_eager_api(self):
        with _test_eager_guard():
            self.test_imperative_api()

    def test_big_dimension(self):
        paddle.disable_static()
        x = paddle.rand(shape=[2, 2, 2, 2, 2, 2, 2])
        np_x = x.numpy()
        z1 = paddle.max(x, axis=-1)
        z2 = paddle.max(x, axis=6)
        np_z1 = z1.numpy()
        np_z2 = z2.numpy()
        z_expected = np.array(np.max(np_x, axis=6))
        self.assertEqual((np_z1 == z_expected).all(), True)
        self.assertEqual((np_z2 == z_expected).all(), True)

    def test_all_negative_axis(self):
        paddle.disable_static()
        x = paddle.rand(shape=[2, 2])
        np_x = x.numpy()
        z1 = paddle.max(x, axis=(-2, -1))
        np_z1 = z1.numpy()
        z_expected = np.array(np.max(np_x, axis=(0, 1)))
        self.assertEqual((np_z1 == z_expected).all(), True)


class TestOutDtype(unittest.TestCase):

    def test_max(self):
        api_fn = paddle.max
        shape = [10, 16]
        check_out_dtype(api_fn,
                        in_specs=[(shape, )],
                        expect_dtypes=['float32', 'float64', 'int32', 'int64'])


class TestMaxWithTensorAxis1(TestReduceOPTensorAxisBase):

    def init_data(self):
        self.pd_api = paddle.max
        self.np_api = np.max
        self.x = paddle.randn([10, 5, 9, 9], dtype='float64')
        self.np_axis = np.array([1, 2], dtype='int64')
        self.tensor_axis = paddle.to_tensor([1, 2], dtype='int64')


class TestMaxWithTensorAxis2(TestReduceOPTensorAxisBase):

    def init_data(self):
        self.pd_api = paddle.max
        self.np_api = np.max
        self.x = paddle.randn([10, 10, 9, 9], dtype='float64')
        self.np_axis = np.array([0, 1, 2], dtype='int64')
        self.tensor_axis = [
            0,
            paddle.to_tensor([1], 'int64'),
            paddle.to_tensor([2], 'int64')
        ]


if __name__ == '__main__':
    unittest.main()

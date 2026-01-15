#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

os.environ['FLAGS_enable_pir_api'] = '0'

import numpy as np
from op_test import get_device_place

import paddle
from paddle import base

paddle.enable_static()


class TestDeg2radAPI(unittest.TestCase):
    def setUp(self):
        self.x_dtype = 'float64'
        self.x_np = np.array(
            [180.0, -180.0, 360.0, -360.0, 90.0, -90.0]
        ).astype(np.float64)
        self.x_shape = [6]
        self.out_np = np.deg2rad(self.x_np)

    def test_static_graph(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='input', dtype=self.x_dtype, shape=self.x_shape
            )
            out = paddle.deg2rad(x)

            place = get_device_place()
            exe = base.Executor(place)
            res = exe.run(
                feed={'input': self.x_np},
                fetch_list=[out],
            )
            np.testing.assert_allclose(
                np.array(res[0]), self.out_np, rtol=1e-05
            )

    def test_dygraph(self):
        paddle.disable_static()
        x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
        result1 = paddle.deg2rad(x1)
        np.testing.assert_allclose(self.out_np, result1.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestDeg2radAPI2(TestDeg2radAPI):
    # Test input data type is int64
    def setUp(self):
        self.x_np = np.array([180]).astype(np.int64)
        self.x_shape = [1]
        self.out_np = np.pi
        self.x_dtype = 'int64'

    def test_dygraph(self):
        paddle.disable_static()

        # Test int64 input
        x2 = paddle.to_tensor([180], dtype="int64")
        result2 = paddle.deg2rad(x2)
        np.testing.assert_allclose(np.pi, result2.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestDeg2radAPI3(TestDeg2radAPI):
    # Test input data type is int32
    def setUp(self):
        self.x_np = np.array([180]).astype(np.int32)
        self.x_shape = [1]
        self.out_np = np.pi
        self.x_dtype = 'int32'

    def test_dygraph(self):
        paddle.disable_static()

        # Test int32 input
        x3 = paddle.to_tensor([180], dtype="int32")
        result3 = paddle.deg2rad(x3)
        np.testing.assert_allclose(np.pi, result3.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestDeg2radAPI4(TestDeg2radAPI):
    # Test input data type is float32
    def setUp(self):
        self.x_np = np.array(
            [180.0, -180.0, 360.0, -360.0, 90.0, -90.0]
        ).astype(np.float32)
        self.x_shape = [6]
        self.out_np = np.deg2rad(self.x_np)
        self.x_dtype = 'float32'


class TestDeg2radAliasAndOut(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor([180.0])
        expected = np.deg2rad(180.0)

        # Test alias
        res = paddle.deg2rad(input=x)
        np.testing.assert_allclose(res.numpy(), expected, rtol=1e-05)

        paddle.enable_static()

    def test_out(self):
        paddle.disable_static()
        x = paddle.to_tensor([180.0])
        expected = np.deg2rad(180.0)

        # Test without out parameter (default None)
        res_no_out = paddle.deg2rad(x)
        np.testing.assert_allclose(res_no_out.numpy(), expected, rtol=1e-05)

        # Test out parameter with float input
        out = paddle.zeros([1], dtype="float32")
        res = paddle.deg2rad(x, out=out)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-05)
        self.assertTrue(res is out)

        # Test out parameter with int64 input
        x_int = paddle.to_tensor([180], dtype="int64")
        out_float = paddle.zeros([1], dtype="float32")
        res = paddle.deg2rad(x_int, out=out_float)
        np.testing.assert_allclose(out_float.numpy(), expected, rtol=1e-05)
        self.assertTrue(res is out_float)

        # Test out parameter with int32 input
        x_int32 = paddle.to_tensor([180], dtype="int32")
        out_float32 = paddle.zeros([1], dtype="float32")
        res = paddle.deg2rad(x_int32, out=out_float32)
        np.testing.assert_allclose(out_float32.numpy(), expected, rtol=1e-05)
        self.assertTrue(res is out_float32)

        paddle.enable_static()


class TestDeg2radStaticOut(unittest.TestCase):
    def test_static_out_float(self):
        """Test out parameter in static graph with float input"""
        paddle.enable_static()
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input', dtype='float32', shape=[1])
            out = paddle.static.data(name='out', dtype='float32', shape=[1])
            result = paddle.deg2rad(x, out=out)

            place = get_device_place()
            exe = base.Executor(place)
            x_np = np.array([180.0]).astype(np.float32)
            out_np = np.zeros([1]).astype(np.float32)
            expected = np.deg2rad(180.0)

            res, out_res = exe.run(
                feed={'input': x_np, 'out': out_np},
                fetch_list=[result, out],
            )
            np.testing.assert_allclose(out_res, expected, rtol=1e-05)

    def test_static_out_int(self):
        """Test out parameter in static graph with int input"""
        paddle.enable_static()
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input', dtype='int64', shape=[1])
            out = paddle.static.data(name='out', dtype='float32', shape=[1])
            result = paddle.deg2rad(x, out=out)

            place = get_device_place()
            exe = base.Executor(place)
            x_np = np.array([180]).astype(np.int64)
            out_np = np.zeros([1]).astype(np.float32)
            expected = np.deg2rad(180.0)

            res, out_res = exe.run(
                feed={'input': x_np, 'out': out_np},
                fetch_list=[result, out],
            )
            np.testing.assert_allclose(out_res, expected, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

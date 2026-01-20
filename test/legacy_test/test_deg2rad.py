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

import unittest

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


class TestDeg2radAlias(unittest.TestCase):
    def test_alias_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor([180.0])
        expected = np.deg2rad(180.0)

        # Test alias with keyword argument
        res = paddle.deg2rad(input=x)
        np.testing.assert_allclose(res.numpy(), expected, rtol=1e-05)

        paddle.enable_static()

    def test_alias_static(self):
        """Test alias parameter in static graph"""
        paddle.enable_static()
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(startup_program, train_program):
            # Test with alias 'input'
            x = paddle.static.data(
                name='input_data', dtype='float32', shape=[1]
            )
            result = paddle.deg2rad(input=x)

            place = get_device_place()
            exe = base.Executor(place)
            x_np = np.array([180.0]).astype(np.float32)
            expected = np.deg2rad(180.0)

            res = exe.run(
                feed={'input_data': x_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(res[0], expected, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

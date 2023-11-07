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

import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()


class TestDeg2radAPI(unittest.TestCase):
    def setUp(self):
        self.x_dtype = 'float64'
        self.x_np = np.array(
            [180.0, -180.0, 360.0, -360.0, 90.0, -90.0]
        ).astype(np.float64)
        self.x_shape = [6]
        self.out_np = np.deg2rad(self.x_np)

    @test_with_pir_api
    def test_static_graph(self):
        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(
                name='input', dtype=self.x_dtype, shape=self.x_shape
            )

            exe = paddle.static.Executor(place)
            out = paddle.deg2rad(x)
            (res,) = exe.run(
                paddle.static.default_main_program(),
                feed={'input': self.x_np},
                fetch_list=[out],
            )
            self.assertTrue((res == self.out_np).all())
        paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static()
        x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
        result1 = paddle.deg2rad(x1)
        np.testing.assert_allclose(self.out_np, result1.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestDeg2radAPI2(TestDeg2radAPI):
    # Test input data type is int
    def setUp(self):
        self.x_np = [180]
        self.x_shape = [1]
        self.out_np = np.pi
        self.x_dtype = 'int64'

    def test_dygraph(self):
        paddle.disable_static()

        x2 = paddle.to_tensor([180])
        result2 = paddle.deg2rad(x2)
        np.testing.assert_allclose(np.pi, result2.numpy(), rtol=1e-05)

        paddle.enable_static()

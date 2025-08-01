# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import utils

import paddle
from paddle.base import core


class CINNConvertToFloat8e4m3Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
    ):
        x_pow2 = x * x
        x_fp8 = x_pow2.astype(paddle.float8_e4m3fn)
        x_fp8.stop_gradient = True
        return x_fp8


class TestFloat2Float8e4m3(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        self.prepare_data()

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.rand(shape=self.shape, dtype=paddle.float32)
        self.x.stop_gradient = True

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNConvertToFloat8e4m3Net()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        core._set_prim_all_enabled(False)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True, use_prim=True)
        dy_out = self.eval(use_cinn=False)

        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-8, rtol=1e-4
        )


class TestBfloat162Float8e4m3(TestFloat2Float8e4m3):
    """
    Test Pir API + @to_static + CINN.
    """

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.clip(
            paddle.randn(self.shape).astype("bfloat16"),
            min=-50,
            max=50,
        )
        self.x.stop_gradient = True


class TestFloat162Float8e4m3(TestFloat2Float8e4m3):
    """
    Test Pir API + @to_static + CINN.
    """

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.rand(shape=self.shape, dtype=paddle.float16)
        self.x.stop_gradient = True


class TestFloat642Float8e4m3(TestFloat2Float8e4m3):
    """
    Test Pir API + @to_static + CINN.
    """

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.rand(shape=self.shape, dtype=paddle.float64)
        self.x.stop_gradient = True


class TestInt322Float8e4m3(TestFloat2Float8e4m3):
    """
    Test Pir API + @to_static + CINN.
    """

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.randint(
            low=-50, high=50, shape=self.shape, dtype=paddle.int32
        )
        self.x.stop_gradient = True


class TestInt642Float8e4m3(TestFloat2Float8e4m3):
    """
    Test Pir API + @to_static + CINN.
    """

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.randint(
            low=-50, high=50, shape=self.shape, dtype=paddle.int64
        )
        self.x.stop_gradient = True


class CINNFloat8e4m3ConvertToNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x_fp8, dtype):
        x_dtype = x_fp8.astype(dtype)
        x_out = x_dtype + x_dtype
        x_out.stop_gradient = True
        return x_out


class TestFloat8e4m32Float(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        self.prepare_data()

    def prepare_data(self):
        self.shape = [8 * 4096, 2048 * 2]
        self.x = paddle.rand(shape=self.shape, dtype=paddle.float8_e4m3fn)
        self.x.stop_gradient = True

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.float32)
        core._set_prim_all_enabled(False)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True, use_prim=True)
        dy_out = self.eval(use_cinn=False)

        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-8, rtol=1e-4
        )


class TestFloat8e4m32Float64(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.float64)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Float16(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.float16)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Bfloat16(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.bfloat16)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Int32(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.int32)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Int64(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.int64)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Int16(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.int16)
        core._set_prim_all_enabled(False)
        return out


class TestFloat8e4m32Int8(TestFloat8e4m32Float):
    """
    Test Pir API + @to_static + CINN.
    """

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = CINNFloat8e4m3ConvertToNet()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, paddle.int8)
        core._set_prim_all_enabled(False)
        return out


if __name__ == '__main__':
    unittest.main()

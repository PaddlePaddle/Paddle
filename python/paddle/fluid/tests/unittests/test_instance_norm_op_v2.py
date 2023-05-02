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
import paddle.nn.functional as F
from paddle import fluid, nn
from paddle.fluid import Program, core, framework, program_guard


class TestInstanceNorm(unittest.TestCase):
    def test_error(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm1d = paddle.nn.InstanceNorm1D(1)
                instance_norm1d(fluid.dygraph.to_variable(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                instance_norm2d = paddle.nn.InstanceNorm2D(1)
                instance_norm2d(fluid.dygraph.to_variable(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(1)
                instance_norm3d(fluid.dygraph.to_variable(x_data_4))

            def weight_bias_false():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(
                    1, weight_attr=False, bias_attr=False
                )

            with fluid.dygraph.guard(p):
                weight_bias_false()
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


def instance_norm_warpper(
    input, weight, bias, epsilon=1e-5, momentum=0.9, data_format='NCHW'
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.instance_norm(
        input, weight, bias, epsilon, momentum, data_format
    )


def _reference_instance_norm(x, scale, bias, epsilon):
    N, C, H, W = x.shape
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    variance = np.var(x, axis=(2, 3), keepdims=True)
    std = np.sqrt(variance) + epsilon
    x_norm = (x - mean) / std
    scale = scale.reshape([1, C, 1, 1])
    bias = bias.reshape([1, C, 1, 1])
    x_norm = scale * x_norm + bias
    return x_norm, mean.reshape(N * C), std.reshape(N * C)


def _reference_instance_norm_grad(x, scale, mean, var):
    n, c, h, w = x.shape
    d_y = np.ones(x.shape) / (np.prod(x.shape))
    d_bias = np.ones((c,)) / c

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    d_scale = np.sum(d_y * (x - mean_tile) * var_tile, axis=(0, 2, 3))
    var_inv = var_tile
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))

    d_x = (
        scale_tile
        * var_inv
        * (
            d_y
            - np.mean(d_y, axis=(2, 3), keepdims=True)
            - (x - mean_tile)
            * var_inv
            * np.mean(
                d_y * (x - mean_tile) * var_inv, axis=(2, 3), keepdims=True
            )
        )
    )

    return d_x, d_scale, d_bias


class TestInstanceNormFP32OP(OpTest):
    def setUp(self):
        '''Test instance_norm op with default value'''
        self.op_type = "instance_norm"
        self.__class__.op_type = self.op_type
        self.python_api = instance_norm_warpper
        self.data_format = "NCHW"
        self.eps = 1e-5
        self.init_dtype()
        self.init_shape()
        self.init_value()
        self.set_err_thre()
        self.inputs = {'X': self.value, 'Scale': self.scale, 'Bias': self.bias}
        self.attrs = {
            'epsilon': self.eps,
            'momentum': 0.9,
            'data_format': self.data_format,
        }
        y, mean, variance = _reference_instance_norm(
            self.value, self.scale, self.bias, self.eps
        )
        self.python_out_sig = ['Y']
        self.outputs = {
            'Y': y,
            'SavedMean': mean,
            'SavedVariance': 1.0 / variance,
        }

    def test_check_output(self):
        self.check_output(atol=self.atol)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Scale', 'Bias'],
            'Y',
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [4, 100, 4, 4]

    def init_value(self):
        np.random.seed(0)
        self.value = np.random.random(self.shape).astype(self.dtype)
        self.scale = np.random.random([self.shape[1]]).astype(np.float32)
        self.bias = np.random.random([self.shape[1]]).astype(np.float32)

    def set_err_thre(self):
        self.atol = 1e-3


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the float16",
)
class TestInstanceNormFP16OP(TestInstanceNormFP32OP):
    def init_dtype(self):
        self.dtype = np.float16

    def set_err_thre(self):
        self.atol = 0.03125
        self.max_relative_error = 8e-3

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=self.atol)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X', 'Scale', 'Bias'],
            'Y',
            max_relative_error=self.max_relative_error,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestInstanceNormBF16OP(OpTest):
    def setUp(self):
        self.op_type = "instance_norm"
        self.__class__.op_type = self.op_type
        self.python_api = instance_norm_warpper
        self.eps = 1e-5
        self.data_format = "NCHW"
        self.dtype = np.uint16
        self.init_shape()
        self.init_value()

        y, mean, variance = _reference_instance_norm(
            self.value, self.scale, self.bias, self.eps
        )
        var_inv = 1.0 / variance
        self.user_defined_grads = _reference_instance_norm_grad(
            self.value, self.scale, mean, var_inv
        )
        self.python_out_sig = ['Y']
        self.outputs = {
            'Y': convert_float_to_uint16(y),
            'SavedMean': mean,
            'SavedVariance': var_inv,
        }
        self.inputs = {
            'X': convert_float_to_uint16(self.value),
            'Scale': self.scale,
            'Bias': self.bias,
        }
        self.attrs = {
            'epsilon': self.eps,
            'momentum': 0.9,
            'data_format': self.data_format,
        }

    def init_value(self):
        np.random.seed(0)
        self.value = np.random.random(self.shape).astype(np.float32)
        self.scale = np.random.random([self.shape[1]]).astype(np.float32)
        self.bias = np.random.random([self.shape[1]]).astype(np.float32)

    def init_shape(self):
        self.shape = [4, 100, 4, 4]

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X', 'Scale', 'Bias'],
            'Y',
            user_defined_grads=self.user_defined_grads,
        )


class PrimNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.instance_norm = nn.InstanceNorm2D(4)

    def forward(self, x):
        y = self.conv(x)
        out = self.instance_norm(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False)


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimNet with @to_static + amp O2(with fp32)
    """

    def setUp(self):
        paddle.seed(2022)
        paddle.disable_static()
        self.x = paddle.randn([4, 2, 6, 6], dtype="float32")
        self.x.stop_gradient = False

    def train(self, use_amp, data_layout="NCHW"):
        paddle.seed(2022)
        net = PrimNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        net = apply_to_static(net, False)
        if use_amp:
            net = paddle.amp.decorate(models=net, level='O2')
        with paddle.amp.auto_cast(enable=use_amp, level='O2'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def test_amp_nchw(self):
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(
                expected,
                actual,
                rtol=1e-3,
                atol=1e-3,
            )


if __name__ == '__main__':
    unittest.main()

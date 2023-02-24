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

import unittest

import numpy as np
from utils import SUB_TOLERANCE

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import core, framework
from paddle.nn import BatchNorm
from paddle.tensor import ones  # noqa: F401
from paddle.tensor import zeros  # noqa: F401

np.random.seed(2023)


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = "float32"
        self.shape = [4, 6, 12, 24]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype
        return

    def set_shape(self, shape) -> None:
        self.shape = shape
        return

    def set_training(self, training) -> None:
        self.training = training
        return

    def set_momentum(self, momentum) -> None:
        self.momentum = momentum
        return

    def set_epsilon(self, epsilon) -> None:
        self.epsilon = epsilon
        return

    def set_data_format(self, data_format) -> None:
        self.data_format = data_format
        return

    def set_use_global_stats(self, use_global_stats) -> None:
        self.use_global_stats = use_global_stats
        return

    def get_rtol(self, flag):
        rtol = SUB_TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = SUB_TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(
    x,
    running_mean,
    running_variance,
    weight,
    bias,
    training,
    momentum,
    epsilon,
    data_format,
    use_global_stats,
):
    z = F.batch_norm(
        x,
        running_mean,
        running_variance,
        weight,
        bias,
        training=training,
        momentum=momentum,
        epsilon=epsilon,
        data_format=data_format,
        use_global_stats=use_global_stats,
    )
    return z


def expect_forward(
    inputs,
    running_mean,
    running_variance,
    weight,
    bias,
    training,
    momentum,
    epsilon,
    data_format,
    use_global_stats,
):
    return fn(
        inputs,
        running_mean,
        running_variance,
        weight,
        bias,
        training,
        momentum,
        epsilon,
        data_format,
        use_global_stats,
    )


class TestCompositeBatchNorm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.training = [False, True]
        self.shapes = [[8, 8, 16, 16], [2, 1, 2, 3]]
        self.momentum = [0.1, 0.9]
        self.data_formats = ["NCHW", "NHWC"]
        self.use_global_stats = [None, True, False]

    def cal_composite(
        self, inputs, running_mean, running_variance, weight, bias
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x1 = paddle.static.data(
                'x1', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x2 = paddle.static.data(
                'x2', shape=running_mean.shape, dtype=str(running_mean.dtype)
            )
            x3 = paddle.static.data(
                'x3',
                shape=running_variance.shape,
                dtype=str(running_variance.dtype),
            )
            x4 = paddle.static.data(
                'x4', shape=weight.shape, dtype=str(weight.dtype)
            )
            x5 = paddle.static.data(
                'x5', shape=bias.shape, dtype=str(bias.dtype)
            )
            y = fn(
                x1,
                x2,
                x3,
                x4,
                x5,
                attrs.training,
                attrs.momentum,
                attrs.epsilon,
                attrs.data_format,
                attrs.use_global_stats,
            )
            blocks = main_program.blocks
            paddle.incubate.autograd.to_prim(blocks)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x1': inputs,
                'x2': running_mean,
                'x3': running_variance,
                'x4': weight,
                'x5': bias,
            },
            fetch_list=[y],
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)

        return res

    def compare_forward(self):
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        if attrs.data_format == 'NCHW':
            C = np_data.shape[1]
        elif attrs.data_format == 'NHWC':
            C = np_data.shape[-1]
        else:
            raise TypeError
        running_mean = paddle.zeros(C, dtype=attrs.dtype)
        running_variance = paddle.ones(C, dtype=attrs.dtype)
        weight = paddle.ones(C, dtype=attrs.dtype) * 2
        bias = paddle.ones(C, dtype=attrs.dtype)

        expect = expect_forward(
            tensor_data,
            running_mean,
            running_variance,
            weight,
            bias,
            attrs.training,
            attrs.momentum,
            attrs.epsilon,
            attrs.data_format,
            attrs.use_global_stats,
        ).numpy()
        np_running_mean = np.zeros(C, dtype=attrs.dtype)
        np_running_variance = np.ones(C, dtype=attrs.dtype)
        np_weight = np.ones(C, dtype=attrs.dtype) * 2
        np_bias = np.ones(C, dtype=attrs.dtype)
        actual = self.cal_composite(
            np_data, np_running_mean, np_running_variance, np_weight, np_bias
        )[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    # def test_forward(self):
    #     for i in self.training:
    #         for j in self.dtypes:
    #             for m in self.momentum:
    #                 attrs.set_training(i)
    #                 attrs.set_dtype(j)
    #                 attrs.set_momentum(m)
    #                 self.compare_forward()

    #     for n in self.shapes:
    #         for s in self.data_formats:
    #             for t in self.use_global_stats:
    #                 attrs.set_shape(n)
    #                 attrs.set_data_format(s)
    #                 attrs.set_use_global_stats(t)
    #                 self.compare_forward()


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super(PrimeNet, self).__init__()
        self.conv = nn.Conv2D(4, 2, (3, 3), bias_attr=False)
        self.bn = BatchNorm(2, act="relu")
        self.run_mean = zeros([2])
        self.run_var = ones([2])
        self.scale = ones([2])
        self.bias = ones([2])

    def forward(self, x):
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([4, 4, 6, 6], dtype="float32")
        self.x.stop_gradient = False

    def train(self, use_prim):
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )

        net = paddle.amp.decorate(models=net, level='O2')

        net = apply_to_static(net, False)
        with paddle.amp.auto_cast(level='O2'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def test_amp(self):
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

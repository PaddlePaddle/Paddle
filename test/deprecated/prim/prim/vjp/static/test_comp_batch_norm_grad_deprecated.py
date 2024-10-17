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

import paddle
import paddle.nn.functional as F
from paddle.base import core, framework

np.random.seed(2023)


class Arg:
    dout = None


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = "float32"
        self.shape = [8, 8, 16, 16]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def set_shape(self, shape) -> None:
        self.shape = shape

    def set_training(self, training) -> None:
        self.training = training

    def set_momentum(self, momentum) -> None:
        self.momentum = momentum

    def set_epsilon(self, epsilon) -> None:
        self.epsilon = epsilon

    def set_data_format(self, data_format) -> None:
        self.data_format = data_format

    def set_use_global_stats(self, use_global_stats) -> None:
        self.use_global_stats = use_global_stats


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
    out = z * paddle.to_tensor(Arg.dout)
    res = paddle.mean(out)
    return res


def expect_grad(
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
    x.stop_gradient = False
    weight.stop_gradient = False
    bias.stop_gradient = False
    res = fn(
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
    )
    gradients = paddle.grad(res, (x, weight, bias))
    return gradients


def cal_composite(inputs, running_mean, running_variance, weight, bias):
    paddle.enable_static()

    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x1 = paddle.static.data(
            'x1', shape=inputs.shape, dtype=str(inputs.dtype)
        )
        x1.stop_gradient = False
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
        x4.stop_gradient = False
        x5 = paddle.static.data('x5', shape=bias.shape, dtype=str(bias.dtype))
        x5.stop_gradient = False
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
        paddle.incubate.autograd.primapi.to_prim(blocks)
        z = paddle.static.gradients([y], [x1, x4, x5])

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
        fetch_list=[z],
    )
    paddle.disable_static()
    return res


class TestCompositeBatchNorm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.training = [False, True]
        self.shapes = [[8, 8, 16, 16], [2, 4, 3, 3]]
        self.momentum = [0.1, 0.9]
        self.epsilon = [1e-05, 2e-05]
        self.data_formats = ["NCHW", "NHWC"]
        self.use_global_stats = [None, True, False]

    def compare_backward(self):
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        Arg.dout = np.random.random(np_data.shape).astype(attrs.dtype)
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

        res_origin = expect_grad(
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
        )
        np_running_mean = np.zeros(C, dtype=attrs.dtype)
        np_running_variance = np.ones(C, dtype=attrs.dtype)
        np_weight = np.ones(C, dtype=attrs.dtype) * 2
        np_bias = np.ones(C, dtype=attrs.dtype)

        res_prim = cal_composite(
            np_data, np_running_mean, np_running_variance, np_weight, np_bias
        )

        vars_name = ["x_grad", "weight_grad", "bias_grad"]
        assert len(res_origin) == len(res_prim)
        for idx in range(len(res_origin)):
            origin_item = res_origin[idx].numpy()
            prim_item = res_prim[idx]
            assert origin_item.dtype == prim_item.dtype
            rtol = 1e-5
            atol = 1e-5
            if (
                not isinstance(
                    framework._current_expected_place(), core.CPUPlace
                )
                and attrs.data_format == "NHWC"
            ):
                rtol = 1e-4
                atol = 1e-4
                if idx in (1, 2):
                    continue

            np.testing.assert_allclose(
                origin_item,
                prim_item,
                rtol=rtol,
                atol=atol,
                err_msg=f"Check diff failed of output: {vars_name[idx]} with data_format: {attrs.data_format}",
            )

    def test_backward_prim_static_vjp(self):
        core._set_prim_backward_enabled(True)
        for i in self.training:
            for j in self.dtypes:
                for k in self.data_formats:
                    for m in self.momentum:
                        attrs.set_training(i)
                        attrs.set_dtype(j)
                        attrs.set_data_format(k)
                        attrs.set_momentum(m)
                        self.compare_backward()

        for s in self.training:
            for n in self.shapes:
                for t in self.use_global_stats:
                    attrs.set_training(s)
                    attrs.set_shape(n)
                    attrs.set_use_global_stats(t)
                    self.compare_backward()
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()

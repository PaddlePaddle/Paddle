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

np.random.seed(2013)

import paddle
import paddle.nn.functional as F

TOLERANCE = {
    "float32": {
        "forward": {"rtol": 1e-6, "atol": 1e-6},
        "backward": {"rtol": 1e-6, "atol": 1e-6},
    },
    "float64": {
        "forward": {"rtol": 1e-7, "atol": 1e-7},
        "backward": {"rtol": 1e-7, "atol": 1e-7},
    },
}


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = "float32"
        self.shape = [2, 1, 2, 3]
        self.training = False  # True leads to error
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
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
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
    # y = paddle.sin(x)
    # x.stop_gradient = False
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
    # res = paddle.cos(z)
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

    gradients = paddle.grad(res, x)
    return gradients


class TestCompositeSoftmax(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32"]
        self.shapes = [[2, 3, 4], [2, 3]]
        self.training = [False, True]
        self.momentum = [0.1]
        self.epsilon = [1e-05, 2e-05]
        self.data_format = ["NCHW", "NHWC"]
        self.use_global_stats = [None, True]

    def cal_composite(
        self, inputs, running_mean, running_variance, weight, bias
    ):
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
            # x4.stop_gradient = False
            x5 = paddle.static.data(
                'x5', shape=bias.shape, dtype=str(bias.dtype)
            )
            # x5.stop_gradient = False
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
            paddle.incubate.autograd.primx.orig2prim()
            # print(blocks)
            paddle.incubate.autograd.primx.prim2orig()
            # print(blocks)
            # paddle.incubate.autograd.to_prim(blocks)

            z = paddle.static.gradients([y], [x1])
            # print(z)
            # names = set()
            # for item in blocks[0].ops:
            #     names.add(item.type)
            # print(sorted(list(names)))
            # # breakpoint()
            # print(blocks)

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

    def compare_backward(self):
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)
        running_mean = paddle.to_tensor([0], dtype="float32")
        running_variance = paddle.to_tensor([1], dtype="float32")
        weight = paddle.to_tensor([2], dtype="float32")
        bias = paddle.to_tensor([1], dtype="float32")

        # expect = expect_forward(
        #     tensor_data, running_mean, running_variance, weight, bias,attrs.training, attrs.momentum, attrs.epsilon, attrs.data_format, attrs.use_global_stats
        # ).numpy()

        expect = expect_grad(
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
        )[0].numpy()

        np_running_mean = np.array([0], dtype="float32")
        np_running_variance = np.array([1], dtype="float32")
        np_weight = np.array([2], dtype="float32")
        np_bias = np.array([1], dtype="float32")

        actual = self.cal_composite(
            np_data, np_running_mean, np_running_variance, np_weight, np_bias
        )[0]
        # breakpoint()
        print("expect=======", expect)
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        # for i in self.training:
        #     for j in self.dtypes:
        #         for m in self.momentum:
        #             for t in self.use_global_stats:
        #                 print("======================= ", i, j, m, t)
        #                 attrs.set_training(i)
        #                 attrs.set_dtype(j)
        #                 attrs.set_momentum(m)
        #                 attrs.set_use_global_stats(t)
        self.compare_backward()


if __name__ == '__main__':
    unittest.main()

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import random
import unittest

import numpy as np
from op_test import get_devices, is_custom_device
from scipy import stats
from utils import dygraph_guard, static_guard

import paddle
from paddle import nn
from paddle.base import Program

DELTA = 0.00001


def _create_random_nd_tensor(dims, size_min, size_max, random_value=False):
    size = [random.randint(size_min, size_max) for _ in range(dims)]
    if random_value:
        tensor = paddle.randn(size)
    else:
        tensor = paddle.zeros(size)
    return tensor


def _random_float(a, b):
    return (b - a) * random.random() + a


def _calculate_gain(nonlinearity, param):
    recommended_gain = {
        'sigmoid': 1,
        'linear': 1,
        'conv1d': 1,
        'conv2d': 1,
        'conv3d': 1,
        'conv1d_transpose': 1,
        'conv_transpose1d': 1,
        'conv2d_transpose': 1,
        'conv_transpose2d': 1,
        'conv3d_transpose': 1,
        'conv_transpose3d': 1,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + param**2)),
        'selu': 3.0 / 4,
    }
    return recommended_gain[nonlinearity]


def _calculate_fan_in_and_fan_out(var: paddle.Tensor) -> tuple[int, int]:
    shape = var.shape
    if not shape or len(shape) == 0:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    return (fan_in, fan_out)


class Test_calculate_gain(unittest.TestCase):
    def test(self):
        for nonlinearity in [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            'conv1d_transpose',
            "conv_transpose1d",
            "conv2d_transpose",
            "conv_transpose2d",
            "conv3d_transpose",
            "conv_transpose3d",
            'sigmoid',
            'tanh',
            "relu",
            "leaky_relu",
            "selu",
        ]:
            self.assertEqual(
                _calculate_gain(nonlinearity, 0),
                paddle.nn.init.calculate_gain(nonlinearity, 0),
            )


class TestCAlFanINOUT(unittest.TestCase):
    def test_cal_fan_in_and_out(self):
        x = paddle.tensor.randn([10])
        self.assertEqual(
            _calculate_fan_in_and_fan_out(x),
            paddle.nn.init._calculate_fan_in_and_fan_out(x),
        )

        y = paddle.tensor.randn([10, 10])
        self.assertEqual(
            _calculate_fan_in_and_fan_out(y),
            paddle.nn.init._calculate_fan_in_and_fan_out(y),
        )

        z = paddle.randn([10, 10, 10])
        self.assertEqual(
            _calculate_fan_in_and_fan_out(z),
            paddle.nn.init._calculate_fan_in_and_fan_out(z),
        )


class Test_kaiming_uniform_(unittest.TestCase):
    def check_kaiming_uniform(
        self, tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'
    ):
        if len(tensor.shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = tensor.shape[0]
            fan_out = tensor.shape[1]
        else:
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]

        if len(tensor.shape) > 2:
            receptive_field_size = np.prod(tensor.shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size

        if mode == "fan_in":
            n = fan_in
        else:
            n = fan_out
        expected_std = _calculate_gain(nonlinearity=nonlinearity, param=a)
        bounds = expected_std * math.sqrt(3.0 / float(n))

        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "uniform", args=(-bounds, bounds * 2))[
            1
        ]
        self.assertGreater(p_value, 0.0001)

    def test_nonlinearity_dygraph(self):
        with dygraph_guard():
            for nonlinearity in [
                'conv_transpose1d',
                'conv_transpose2d',
                'conv_transpose3d',
                'relu',
                'leaky_relu',
            ]:
                input_tensor = paddle.zeros([1024, 512])
                paddle.nn.init.kaiming_uniform_(
                    input_tensor, nonlinearity=nonlinearity
                )
                self.check_kaiming_uniform(
                    input_tensor, nonlinearity=nonlinearity
                )

    def test_dygraph(self):
        with dygraph_guard():
            for use_a in [True, False]:
                for dims in [2, 3, 4]:
                    for mode in ["fan_in", "fan_out"]:
                        input_tensor = _create_random_nd_tensor(
                            dims, size_min=20, size_max=108
                        )
                        if use_a:
                            a = _random_float(0.1, 2)
                        else:
                            a = 0
                        paddle.nn.init.kaiming_uniform_(
                            input_tensor, a=a, mode=mode
                        )
                        self.check_kaiming_uniform(input_tensor, a=a, mode=mode)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.kaiming_uniform_
            init(linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            self.check_kaiming_uniform(
                linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
            )

            init(
                linear.weight, a=-0.2, mode="fan_out", nonlinearity="leaky_relu"
            )
            self.check_kaiming_uniform(
                linear.weight, a=-0.2, mode="fan_out", nonlinearity="leaky_relu"
            )

            init(linear.weight, a=0, mode="fan_in", nonlinearity="relu")
            self.check_kaiming_uniform(
                linear.weight, a=0, mode="fan_in", nonlinearity="relu"
            )

            init(linear.weight, a=0, mode="fan_out", nonlinearity="relu")
            self.check_kaiming_uniform(
                linear.weight, a=0, mode="fan_out", nonlinearity="relu"
            )

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_kaiming_uniform_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.kaiming_uniform_(input_tensor)
            self.check_kaiming_uniform(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.kaiming_uniform_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check_kaiming_uniform(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.kaiming_uniform_(
                        x, a=0.1, mode='fan_out'
                    )
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check_kaiming_uniform(pd_res, a=0.1, mode='fan_out')


class Test_kaiming_normal_(unittest.TestCase):
    def check_kaiming_normal(
        self, tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'
    ):
        if len(tensor.shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = tensor.shape[0]
            fan_out = tensor.shape[1]
        else:
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]

        if len(tensor.shape) > 2:
            receptive_field_size = np.prod(tensor.shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size

        if mode == "fan_in":
            n = fan_in
        else:
            n = fan_out
        expected_std = _calculate_gain(nonlinearity=nonlinearity, param=a)
        std = expected_std / math.sqrt(float(n))

        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "norm", args=(0.0, std))[1]
        self.assertGreater(p_value, 0.0001)

    def test_nonlinearity_dygraph(self):
        with dygraph_guard():
            for nonlinearity in [
                'conv_transpose1d',
                'conv_transpose2d',
                'conv_transpose3d',
                'relu',
                'leaky_relu',
            ]:
                input_tensor = paddle.zeros([1024, 512])
                paddle.nn.init.kaiming_normal_(
                    input_tensor, nonlinearity=nonlinearity
                )
                self.check_kaiming_normal(
                    input_tensor, nonlinearity=nonlinearity
                )

    def test_dygraph(self):
        with dygraph_guard():
            for use_a in [True, False]:
                for dims in [2, 3, 4]:
                    for mode in ["fan_in", "fan_out"]:
                        input_tensor = _create_random_nd_tensor(
                            dims, size_min=20, size_max=108
                        )
                        if use_a:
                            a = _random_float(0.1, 2)
                        else:
                            a = 0
                        paddle.nn.init.kaiming_normal_(
                            input_tensor, a=a, mode=mode
                        )
                        self.check_kaiming_normal(input_tensor, a=a, mode=mode)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.kaiming_normal_
            init(linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            self.check_kaiming_normal(
                linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
            )

            init(
                linear.weight, a=-0.2, mode="fan_out", nonlinearity="leaky_relu"
            )
            self.check_kaiming_normal(
                linear.weight, a=-0.2, mode="fan_out", nonlinearity="leaky_relu"
            )

            init(linear.weight, a=0, mode="fan_in", nonlinearity="relu")
            self.check_kaiming_normal(
                linear.weight, a=0, mode="fan_in", nonlinearity="relu"
            )

            init(linear.weight, a=0, mode="fan_out", nonlinearity="relu")
            self.check_kaiming_normal(
                linear.weight, a=0, mode="fan_out", nonlinearity="relu"
            )

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.kaiming_normal_(input_tensor)
            self.check_kaiming_normal(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.kaiming_normal_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check_kaiming_normal(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.kaiming_normal_(
                        x, a=0.1, mode='fan_out'
                    )
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check_kaiming_normal(pd_res, a=0.1, mode='fan_out')


class Test_xavier_uniform_(unittest.TestCase):
    def check(self, tensor, gain=1.0):
        if len(tensor.shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = tensor.shape[0]
            fan_out = tensor.shape[1]
        else:
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]

        if len(tensor.shape) > 2:
            receptive_field_size = np.prod(tensor.shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size

        bounds = gain * math.sqrt(6.0 / float(fan_in + fan_out))

        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "uniform", args=(-bounds, bounds * 2))[
            1
        ]
        self.assertGreater(p_value, 0.0001)

    def test_dygraph(self):
        with dygraph_guard():
            for use_gain in [True, False]:
                for dims in [2, 3, 4]:
                    input_tensor = _create_random_nd_tensor(
                        dims, size_min=20, size_max=108
                    )
                    if use_gain:
                        gain = _random_float(0.1, 3.0)
                    else:
                        gain = 1.0
                    paddle.nn.init.xavier_uniform_(input_tensor, gain=gain)
                    self.check(input_tensor, gain=gain)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.xavier_uniform_
            init(linear.weight, gain=0.2)
            self.check(linear.weight, gain=0.2)

            init(linear.weight, gain=0.25)
            self.check(linear.weight, gain=0.25)

            init(linear.weight, gain=1.0)
            self.check(linear.weight, gain=1.0)

            init(linear.weight, gain=2.0)
            self.check(linear.weight, gain=2.0)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.xavier_uniform_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.xavier_uniform_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.xavier_uniform_(x, gain=0.5)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, gain=0.5)


class Test_xavier_normal_(unittest.TestCase):
    def check(self, tensor, gain=1.0):
        if len(tensor.shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = tensor.shape[0]
            fan_out = tensor.shape[1]
        else:
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]

        if len(tensor.shape) > 2:
            receptive_field_size = np.prod(tensor.shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size

        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "norm", args=(0.0, std))[1]
        self.assertGreater(p_value, 0.0001)

    def test_dygraph(self):
        with dygraph_guard():
            for use_gain in [True, False]:
                for dims in [2, 3, 4]:
                    input_tensor = _create_random_nd_tensor(
                        dims, size_min=20, size_max=108
                    )
                    if use_gain:
                        gain = _random_float(0.1, 3.0)
                    else:
                        gain = 1.0
                    paddle.nn.init.xavier_normal_(input_tensor, gain=gain)
                    self.check(input_tensor, gain=gain)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.xavier_normal_
            init(linear.weight, gain=1.0)
            self.check(linear.weight, gain=1.0)

            init(linear.weight, gain=2.6)
            self.check(linear.weight, gain=2.6)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.xavier_normal_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.xavier_normal_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.xavier_normal_(x, gain=0.3)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, gain=0.3)


class Test_uniform_(unittest.TestCase):
    def check(self, tensor, a=0.0, b=1.0):
        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        self.assertGreater(p_value, 0.0001)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.uniform_
            init(linear.weight, a=0.2, b=1.3)
            self.check(linear.weight, a=0.2, b=1.3)

            init(linear.weight, a=2.2, b=4.3)
            self.check(linear.weight, a=2.2, b=4.3)
            init(linear.weight, a=-0.2, b=0.2)
            self.check(linear.weight, a=-0.2, b=0.2)
            init(linear.weight, a=-1.5, b=1.5)
            self.check(linear.weight, a=-1.5, b=1.5)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                paddle.nn.init.uniform_(input_tensor, a=-3.0, b=2.0)
                self.check(input_tensor, -3.0, 2.0)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.uniform_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.uniform_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.uniform_(x, a=0.4, b=1.9)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, a=0.4, b=1.9)


class Test_normal_(unittest.TestCase):
    def check(self, tensor, mean=0.0, std=1.0):
        samples = tensor.flatten().tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        self.assertGreater(p_value, 0.0001)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.normal_
            init(linear.weight, mean=0.2, std=1.3)
            self.check(linear.weight, mean=0.2, std=1.3)

            init(linear.weight, mean=2.2, std=4.3)
            self.check(linear.weight, mean=2.2, std=4.3)
            init(linear.weight, mean=-0.2, std=0.2)
            self.check(linear.weight, mean=-0.2, std=0.2)
            init(linear.weight, mean=-1.5, std=1.5)
            self.check(linear.weight, mean=-1.5, std=1.5)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                mean = _random_float(-3.0, 3.0)
                std = _random_float(0.5, 3.0)
                paddle.nn.init.normal_(input_tensor, mean, std)
                self.check(input_tensor, mean, std)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.normal_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.normal_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.normal_(x, mean=0.4, std=1.9)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, mean=0.4, std=1.9)


class Test_trunc_normal_(unittest.TestCase):
    def check(self, tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        samples = ((tensor.flatten() - mean) / std).tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(samples, "truncnorm", args=(a0, b0))[1]
        self.assertGreater(p_value, 0.0001)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.trunc_normal_
            init(linear.weight, mean=0.2, std=1.3, a=1.0, b=2.0)
            self.check(linear.weight, mean=0.2, std=1.3, a=1.0, b=2.0)

            init(linear.weight, mean=2.2, std=4.3, a=1.3, b=2.0)
            self.check(linear.weight, mean=2.2, std=4.3, a=1.3, b=2.0)
            init(linear.weight, mean=-0.2, std=0.2, a=-1.0, b=2.9)
            self.check(linear.weight, mean=-0.2, std=0.2, a=-1.0, b=2.9)
            init(linear.weight, mean=-1.5, std=1.5, a=-1.4, b=2.9)
            self.check(linear.weight, mean=-1.5, std=1.5, a=-1.4, b=2.9)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                mean = _random_float(-3.0, 3.0)
                std = _random_float(0.5, 3.0)
                bound = _random_float(0.5, 10)
                a = mean - bound
                b = mean + bound
                paddle.nn.init.trunc_normal_(input_tensor, mean, std, a, b)
                self.check(input_tensor, mean, std, a, b)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.trunc_normal_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.trunc_normal_(
                        x, mean=0.4, std=1.9, a=-1.9, b=6
                    )
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, mean=0.4, std=1.9, a=-1.9, b=6)


class Test_constant_(unittest.TestCase):
    def check(self, tensor, val):
        if isinstance(tensor, paddle.Tensor):
            diff = (tensor - val).abs().max().item()
        elif isinstance(tensor, np.ndarray):
            diff = np.max(np.abs(tensor - val))
        self.assertLess(diff, 0.000001)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.constant_
            init(linear.weight, val=1.0)
            self.check(linear.weight, val=1.0)

            init(linear.weight, val=0.8)
            self.check(linear.weight, val=0.8)
            init(linear.weight, val=0.0)
            self.check(linear.weight, val=0.0)
            init(linear.weight, val=1.9)
            self.check(linear.weight, val=1.9)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                val = _random_float(-1024.0, 1024.0)
                paddle.nn.init.constant_(input_tensor, val)
                self.check(input_tensor, val)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.constant_(x, val=-0.4)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, val=-0.4)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.constant_(x, val=8.4)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res, val=8.4)


class Test_ones_(unittest.TestCase):
    def check(self, tensor, eps=1e-6):
        if isinstance(tensor, paddle.Tensor):
            diff = (tensor - 1.0).abs().max().item()
        elif isinstance(tensor, np.ndarray):
            diff = np.max(np.abs(tensor - 1.0))
        self.assertLess(diff, eps)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.ones_
            init(linear.weight)
            self.check(linear.weight)

            init(linear.weight)
            self.check(linear.weight)
            init(linear.weight)
            self.check(linear.weight)
            init(linear.weight)
            self.check(linear.weight)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                paddle.nn.init.ones_(input_tensor)
                self.check(input_tensor)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.ones_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.ones_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.ones_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16


class Test_zeros_(unittest.TestCase):
    def check(self, tensor, eps=1e-6):
        if isinstance(tensor, paddle.Tensor):
            diff = tensor.abs().max().item()
        elif isinstance(tensor, np.ndarray):
            diff = np.max(np.abs(tensor))
        self.assertLess(diff, eps)

    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.zeros_
            init(linear.weight)
            self.check(linear.weight)

            init(linear.weight)
            self.check(linear.weight)
            init(linear.weight)
            self.check(linear.weight)
            init(linear.weight)
            self.check(linear.weight)

    def test_dygraph(self):
        with dygraph_guard():
            for dims in [2, 3, 4]:
                input_tensor = _create_random_nd_tensor(
                    dims, size_min=20, size_max=108
                )
                paddle.nn.init.zeros_(input_tensor)
                self.check(input_tensor)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.zeros_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    def test_static_graph_case2(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([100, 52, 3, 4]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[100, 52, 3, 4], dtype='float32'
                    )
                    out = paddle.nn.init.zeros_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([1024, 512], dtype='float16')
            paddle.nn.init.zeros_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16


class Test_eye_(unittest.TestCase):
    def check(self, tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()
        row, col = tensor.shape
        expected = np.eye(row, col)
        self.assertEqual((tensor == expected).all(), True)

    @unittest.skipIf(
        paddle.base.is_compiled_with_rocm(), "ROCM does not support this API"
    )
    def test_linear_dygraph(self):
        with dygraph_guard():
            linear = nn.Linear(40, 20)
            init = paddle.nn.init.eye_
            init(linear.weight)
            self.check(linear.weight)

    @unittest.skipIf(
        paddle.base.is_compiled_with_rocm(), "ROCM does not support this API"
    )
    def test_dygraph(self):
        with dygraph_guard():
            input_tensor = _create_random_nd_tensor(
                2, size_min=20, size_max=108
            )
            paddle.nn.init.eye_(input_tensor)
            self.check(input_tensor)

    @unittest.skipIf(
        paddle.base.is_compiled_with_rocm(), "ROCM does not support this API"
    )
    def test_dims_error(self):
        with dygraph_guard():
            with self.assertRaises(AssertionError):
                input_tensor = paddle.zeros([5, 5, 1024, 512, 10, 2])
                paddle.nn.init.eye_(input_tensor)
            with self.assertRaises(AssertionError):
                input_tensor = paddle.zeros([5, 5, 4])
                paddle.nn.init.eye_(input_tensor)

    @unittest.skipIf(
        paddle.base.is_compiled_with_rocm(), "ROCM does not support this API"
    )
    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.eye_(x)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]
                    self.check(pd_res)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([128, 64], dtype='float16')
            paddle.nn.init.eye_(input_tensor)
            self.check(input_tensor)
            assert input_tensor.dtype == paddle.float16


class Test_dirac_(unittest.TestCase):
    def test_dygraph(self):
        with dygraph_guard():
            for dims in [3, 4, 5]:
                for groups in [1, 2, 3]:
                    a, c, d, e = (random.randint(1, 5) for _ in range(4))
                    b = random.randint(1, 5 * groups)
                    input_tensor = paddle.randn((a * groups, b, c, d, e)[:dims])

                    paddle.nn.init.dirac_(input_tensor, groups)

                    c_out, c_in = (
                        input_tensor.shape[0] // groups,
                        input_tensor.shape[1],
                    )
                    min_d = min(c_out, c_in)
                    assert (
                        paddle.nonzero(input_tensor).shape[0] == min_d * groups
                    )
                    self.assertEqual(input_tensor.sum(), min_d * groups)

    def test_dims_error(self):
        with dygraph_guard():
            with self.assertRaises(AssertionError):
                input_tensor = paddle.zeros([5, 5, 1024, 512, 10, 2])
                paddle.nn.init.dirac_(input_tensor)
            with self.assertRaises(AssertionError):
                input_tensor = paddle.zeros([5, 5])
                paddle.nn.init.dirac_(input_tensor)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5, 20]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5, 20], dtype='float32'
                    )
                    out = paddle.nn.init.dirac_(x, groups=2)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]

                    c_out, c_in = pd_res.shape[0] // 2, pd_res.shape[1]
                    min_d = min(c_out, c_in)
                    assert np.nonzero(pd_res)[0].shape[0] == min_d * 2
                    self.assertEqual(pd_res.sum(), min_d * 2)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_fp16(self):
        with dygraph_guard():
            input_tensor = paddle.zeros([5, 5, 1024, 512], dtype='float16')
            paddle.nn.init.dirac_(input_tensor)
            assert input_tensor.dtype == paddle.float16


class Test_orthogonal_(unittest.TestCase):
    def check(self, tensor, gain):
        if isinstance(tensor, paddle.Tensor):
            tensor = tensor.numpy()

        tensor = tensor.reshape([tensor.shape[0], -1])

        row, col = tensor.shape
        if row > col:
            np.testing.assert_allclose(
                gain**2 * np.eye(col),
                np.matmul(tensor.T, tensor),
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            np.testing.assert_allclose(
                gain**2 * np.eye(row),
                np.matmul(tensor, tensor.T),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_dygraph(self):
        with dygraph_guard():
            for use_gain in [True, False]:
                for tensor_size in [
                    [3, 4],
                    [4, 3],
                    [20, 2, 3, 4],
                    [2, 3, 4, 5],
                ]:
                    input_tensor = paddle.zeros(tensor_size)
                    gain = 1.0

                    if use_gain:
                        gain = _random_float(0.1, 2)

                    paddle.nn.init.orthogonal_(input_tensor, gain=gain)

                    self.check(input_tensor, gain=gain)

    def test_dims_error(self):
        with dygraph_guard(), self.assertRaises(AssertionError):
            input_tensor = paddle.zeros(
                [
                    5,
                ]
            )
            paddle.nn.init.orthogonal_(input_tensor)

    def test_static_graph_case1(self):
        self.place = get_devices()
        with static_guard():
            for place in self.place:
                x_np = np.zeros([10, 5]).astype('float32')
                with paddle.static.program_guard(Program()):
                    x = paddle.static.data(
                        name="x", shape=[10, 5], dtype='float32'
                    )
                    out = paddle.nn.init.orthogonal_(x, gain=0.4)
                    exe = paddle.static.Executor(place=place)
                    feed_list = {"x": x_np}
                    pd_res = exe.run(
                        paddle.static.default_main_program(),
                        feed=feed_list,
                        fetch_list=[out],
                    )[0]

                    self.check(pd_res, gain=0.4)


if __name__ == '__main__':
    unittest.main()

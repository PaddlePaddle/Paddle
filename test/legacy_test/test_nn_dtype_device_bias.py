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
import re
import unittest

import numpy as np
from op_test import get_device_place, is_custom_device
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, nn


def convert_place_to_device(place):
    re_exp = re.compile(r'[(](.+?)[)]', re.DOTALL)
    place_str = re.findall(re_exp, str(place))[0]
    return place_str


def devices_and_type():
    devices = {paddle.CPUPlace(): 0, "cpu": 0}
    if paddle.device.is_compiled_with_cuda() or is_custom_device():
        # 1 means cuda place, see paddle/phi/kernels/memcpy_kernel.cc
        devices[get_device_place()] = 1
        devices['gpu:0'] = 1
    if paddle.device.is_compiled_with_xpu():
        devices[paddle.device.XPUPlace(0)] = 3
    if paddle.device.is_compiled_with_ipu():
        devices[paddle.device.IPUPlace()] = 4
    return devices


def check_dtype_device(tensor, dtype, device):
    if isinstance(dtype, str):
        assert tensor.dtype == getattr(paddle, dtype), (
            f"expect {dtype}, but got {tensor.dtype}"
        )
    else:
        assert tensor.dtype == dtype, f"expect {dtype}, but got {tensor.dtype}"

    place = convert_place_to_device(tensor.place)
    if not isinstance(device, str):
        device = convert_place_to_device(device)
    assert place == device, f"expect {device}, but got {place}"


class Test_Conv3D(unittest.TestCase):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv3D

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12, 12, 12], dtype=dtype).to(device)
            conv = self.api(8, 16, 3, dtype=dtype, device=device)
            check_dtype_device(conv.weight, dtype, device)
            check_dtype_device(conv.bias, dtype, device)

            y_var = conv(x_var)
            check_dtype_device(y_var, dtype, device)

            # check "input"
            y_var = conv(input=x_var)
            check_dtype_device(y_var, dtype, device)

            # check "x"
            y_var = conv(x=x_var)
            check_dtype_device(y_var, dtype, device)

    def test_dygraph(self):
        for dtype in self.dtypes:
            for device, _ in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_dygraph_one(dtype=dtype, device=device)

    def run_test_static_one(self, dtype, device, dst_place_type):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1, -1, -1)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                conv = self.api(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    dtype=dtype,
                    device=device,
                )
                # check "input"
                y_var = conv(input=x_var)
                # check "x"
                y_var = conv(x=x_var)
            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randn(5, 8, 12, 12, 12).astype(dtype_str)

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert op.attrs()['dst_place_type'] == dst_place_type, (
                        f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"
                    )

    def test_static(self):
        for dtype in self.dtypes:
            for device, dst_place_type in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_static_one(
                        dtype=dtype,
                        device=device,
                        dst_place_type=dst_place_type,
                    )

    def test_bias_dygraph(self):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12, 12, 12])
            conv = self.api(8, 16, 3, bias=True)
            y_var = conv(x_var)
            assert isinstance(conv.bias, paddle.Tensor)

            conv = self.api(8, 16, 3, bias=False, bias_attr=True)
            y_var = conv(x_var)
            assert conv.bias is None

    def test_bias_static(self):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1, -1, -1)

                x_var = paddle.static.data("input", input_shape)
                conv = self.api(8, 16, 3, bias=False)
                y_var = conv(x_var)
                assert conv.bias is None

            feed_dict = {
                "input": np.random.randn(5, 8, 12, 12, 12).astype('float32')
            }
            exe = base.Executor()
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])


class Test_Conv3d(Test_Conv3D):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv3d


class Test_Conv2D(unittest.TestCase):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv2D

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12, 12], dtype=dtype).to(device)
            conv = self.api(8, 16, 3, dtype=dtype, device=device)
            check_dtype_device(conv.weight, dtype, device)
            check_dtype_device(conv.bias, dtype, device)

            y_var = conv(x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = conv(input=x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = conv(x=x_var)
            check_dtype_device(y_var, dtype, device)

    def test_dygraph(self):
        for dtype in self.dtypes:
            for device, _ in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_dygraph_one(dtype=dtype, device=device)

    def run_test_static_one(self, dtype, device, dst_place_type):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1, -1)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                conv = self.api(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    dtype=dtype,
                    device=device,
                )
                y_var = conv(x_var)
                y_var = conv(input=x_var)

            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randn(5, 8, 12, 12).astype(dtype_str)

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert op.attrs()['dst_place_type'] == dst_place_type, (
                        f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"
                    )

    def test_static(self):
        for dtype in self.dtypes:
            for device, dst_place_type in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_static_one(
                        dtype=dtype,
                        device=device,
                        dst_place_type=dst_place_type,
                    )

    def test_bias_dygraph(self):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12, 12])
            conv = self.api(8, 16, 3, bias=True)
            y_var = conv(x_var)
            assert isinstance(conv.bias, paddle.Tensor)

            conv = self.api(8, 16, 3, bias=False)
            y_var = conv(x_var)
            assert conv.bias is None

    def test_bias_static(self):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1, -1)

                x_var = paddle.static.data("input", input_shape)
                conv = self.api(8, 16, 3, bias=False)
                y_var = conv(x_var)
                assert conv.bias is None

            feed_dict = {
                "input": np.random.randn(5, 8, 12, 12).astype('float32')
            }
            exe = base.Executor()
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])


class Test_Conv2d(Test_Conv2D):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv2d


class Test_Conv1D(unittest.TestCase):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv1D

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12], dtype=dtype).to(device)
            conv = self.api(8, 16, 3, dtype=dtype, device=device)
            check_dtype_device(conv.weight, dtype, device)
            check_dtype_device(conv.bias, dtype, device)

            y_var = conv(x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = conv(input=x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = conv(x=x_var)
            check_dtype_device(y_var, dtype, device)

    def test_dygraph(self):
        for dtype in self.dtypes:
            for device, _ in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_dygraph_one(dtype=dtype, device=device)

    def run_test_static_one(self, dtype, device, dst_place_type):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                conv = self.api(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    dtype=dtype,
                    device=device,
                )
                y_var = conv(x_var)
                y_var = conv(input=x_var)

            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randn(5, 8, 12).astype(dtype_str)

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert op.attrs()['dst_place_type'] == dst_place_type, (
                        f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"
                    )

    def test_static(self):
        for dtype in self.dtypes:
            for device, dst_place_type in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_static_one(
                        dtype=dtype,
                        device=device,
                        dst_place_type=dst_place_type,
                    )

    def test_bias_dygraph(self):
        with dygraph_guard():
            x_var = paddle.randn([5, 8, 12])
            conv = self.api(8, 16, 3, bias=True)
            y_var = conv(x_var)
            assert isinstance(conv.bias, paddle.Tensor)

            conv = self.api(8, 16, 3, bias=False)
            y_var = conv(x_var)
            assert conv.bias is None

    def test_bias_static(self):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1, 8, -1)

                x_var = paddle.static.data("input", input_shape)
                conv = self.api(8, 16, 3, bias=False)
                y_var = conv(x_var)
                assert conv.bias is None

            feed_dict = {"input": np.random.randn(5, 8, 12).astype('float32')}
            exe = base.Executor()
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])


class Test_Conv1d(Test_Conv1D):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Conv1d


class Test_Embedding(unittest.TestCase):
    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'
        self.api = nn.Embedding

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randint(low=0, high=32, shape=[128]).to(device)
            layer = self.api(32, 16, dtype=dtype, device=device)
            check_dtype_device(layer.weight, dtype, device)

            y_var = layer(x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = layer(input=x_var)
            check_dtype_device(y_var, dtype, device)

            y_var = layer(x=x_var)
            check_dtype_device(y_var, dtype, device)

    def test_dygraph(self):
        for dtype in self.dtypes:
            for device, _ in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_dygraph_one(dtype=dtype, device=device)

    def run_test_static_one(self, dtype, device, dst_place_type):
        with static_guard():
            main = base.Program()
            start = base.Program()
            with (
                base.unique_name.guard(),
                base.program_guard(main, start),
            ):
                input_shape = (-1,)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                layer = self.api(
                    32,
                    16,
                    dtype=dtype,
                    device=device,
                )
                y_var = layer(x_var)
                y_var = layer(input=x_var)

            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randint(0, 32, size=(128,))

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert op.attrs()['dst_place_type'] == dst_place_type, (
                        f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"
                    )

    def test_static(self):
        for dtype in self.dtypes:
            for device, dst_place_type in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_static_one(
                        dtype=dtype,
                        device=device,
                        dst_place_type=dst_place_type,
                    )

    def test_weight_freeze(self):
        with dygraph_guard():
            x_var = paddle.randint(low=0, high=32, shape=[128])
            weight = paddle.randn([32, 16])
            layer = self.api(32, 16, _weight=weight, _freeze=True)

            y_var = layer(x_var)
            np.testing.assert_allclose(weight.numpy(), layer.weight.numpy())
            np.testing.assert_allclose(
                y_var.numpy(),
                paddle.nn.functional.one_hot(x_var, num_classes=32).numpy()
                @ weight.numpy(),
            )
            assert layer.weight.stop_gradient

    def test_padding_idx(self):
        with dygraph_guard():
            layer = self.api(32, 16, padding_idx=2)
            assert layer._padding_idx == layer.padding_idx

            layer.padding_idx = 5
            assert layer._padding_idx == 5


if __name__ == '__main__':
    unittest.main()

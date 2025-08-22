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
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, nn


def convert_place_to_device(place):
    re_exp = re.compile(r'[(](.+?)[)]', re.DOTALL)
    place_str = re.findall(re_exp, str(place))[0]
    return place_str


def devices_and_type():
    devices = {paddle.CPUPlace(): 0, "cpu": 0}
    if paddle.device.is_compiled_with_cuda():
        # 1 means cuda place, see paddle/phi/kernels/memcpy_kernel.cc
        devices[paddle.CUDAPlace(0)] = 1
        devices['gpu:0'] = 1
    if paddle.device.is_compiled_with_xpu():
        devices[paddle.device.XPUPlace(0)] = 3
    if paddle.device.is_compiled_with_ipu():
        devices[paddle.device.IPUPlace()] = 4
    return devices


def check_dtype_device(tensor, dtype, device):
    if isinstance(dtype, str):
        assert tensor.dtype == getattr(
            paddle, dtype
        ), f"expect {dtype}, but got {tensor.dtype}"
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

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([10, 16, 32, 32, 32], dtype=dtype).to(device)
            conv = nn.Conv3D(16, 33, 3, dtype=dtype, device=device)
            check_dtype_device(conv.weight, dtype, device)
            check_dtype_device(conv.bias, dtype, device)

            y_var = conv(x_var)
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
                input_shape = (-1, 16, -1, -1, -1)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                conv = paddle.nn.Conv3D(
                    in_channels=16,
                    out_channels=33,
                    kernel_size=3,
                    dtype=dtype,
                    device=device,
                )
                y_var = conv(x_var)
            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randn(10, 16, 32, 32, 32).astype(dtype_str)

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert (
                        op.attrs()['dst_place_type'] == dst_place_type
                    ), f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"

    def test_static(self):
        for dtype in self.dtypes:
            for device, dst_place_type in self.devices.items():
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_static_one(
                        dtype=dtype,
                        device=device,
                        dst_place_type=dst_place_type,
                    )


class Test_Conv3d(unittest.TestCase):

    def setUp(self):
        self.devices = devices_and_type()
        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]
        self.op_name = 'pd_op.memcpy'

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([10, 16, 32, 32, 32], dtype=dtype).to(device)
            conv = nn.Conv3d(16, 33, 3, dtype=dtype, device=device)
            check_dtype_device(conv.weight, dtype, device)
            check_dtype_device(conv.bias, dtype, device)

            y_var = conv(x_var)
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
                input_shape = (-1, 16, -1, -1, -1)

                x_var = paddle.static.data("input", input_shape, dtype=dtype)
                conv = paddle.nn.Conv3d(
                    in_channels=16,
                    out_channels=33,
                    kernel_size=3,
                    dtype=dtype,
                    device=device,
                )
                y_var = conv(x_var)
            if isinstance(dtype, str):
                dtype_str = dtype
            else:
                dtype_str = str(dtype).replace('paddle.', '')
            input = np.random.randn(10, 16, 32, 32, 32).astype(dtype_str)

            feed_dict = {"input": input}
            exe = base.Executor(device)
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
            assert y_np.dtype == dtype_str
            for op in main.global_block().ops:
                if op.name() == self.op_name:
                    assert (
                        op.attrs()['dst_place_type'] == dst_place_type
                    ), f"expect {dst_place_type}, but got {op.attrs()['dst_place_type']}"

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
            x_var = paddle.randn([10, 16, 32, 32, 32])
            conv = nn.Conv3d(16, 33, 3, bias=True)
            y_var = conv(x_var)
            assert isinstance(conv.bias, paddle.Tensor)

            conv = nn.Conv3d(16, 33, 3, bias=False)
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
                input_shape = (-1, 16, -1, -1, -1)

                x_var = paddle.static.data("input", input_shape)
                conv = nn.Conv3d(16, 33, 3, bias=False)
                y_var = conv(x_var)
                assert conv.bias is None

            feed_dict = {
                "input": np.random.randn(10, 16, 32, 32, 32).astype('float32')
            }
            exe = base.Executor()
            exe.run(start)
            (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])


if __name__ == '__main__':
    unittest.main()

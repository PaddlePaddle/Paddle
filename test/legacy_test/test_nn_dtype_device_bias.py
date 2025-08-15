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


class Test_Conv3D(unittest.TestCase):

    def check(self, tensor, dtype, device):
        if isinstance(dtype, str):
            assert tensor.dtype == getattr(
                paddle, dtype
            ), f"expect {dtype}, but got {tensor.dtype}"
        else:
            assert (
                tensor.dtype == dtype
            ), f"expect {dtype}, but got {tensor.dtype}"

        place = convert_place_to_device(tensor.place)
        if not isinstance(device, str):
            device = convert_place_to_device(device)
        assert place == device, f"expect {device}, but got {place}"

    def setUp(self):
        self.devices = [paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda():
            count = paddle.device.cuda.device_count()
            self.devices.extend([f"gpu:{i}" for i in range(count)])
            self.devices.extend([paddle.CUDAPlace(i) for i in range(count)])
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.device.XPUPlace(0))
        if paddle.device.is_compiled_with_ipu():
            self.devices.append(paddle.device.IPUPlace())

        self.dtypes = ["float32", paddle.float32, 'float64', paddle.float64]

    def run_test_dygraph_one(self, dtype, device):
        with dygraph_guard():
            x_var = paddle.randn([10, 16, 32, 32, 32], dtype=dtype).to(device)
            conv = nn.Conv3D(16, 33, 3, dtype=dtype, device=device)
            self.check(conv.weight, dtype, device)
            self.check(conv.bias, dtype, device)

            y_var = conv(x_var)
            self.check(y_var, dtype, device)

    def test_dygraph(self):
        for dtype in self.dtypes:
            for device in self.devices:
                with self.subTest(msg=f"Testing {dtype} on {device}"):
                    self.run_test_dygraph_one(dtype=dtype, device=device)

    def test_bias_dygraph(self):
        with dygraph_guard():
            x_var = paddle.randn([10, 16, 32, 32, 32])
            conv = nn.Conv3D(16, 33, 3, bias=True)
            y_var = conv(x_var)
            assert isinstance(conv.bias, paddle.Tensor)

            conv = nn.Conv3D(16, 33, 3, bias=False)
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
                conv = nn.Conv3D(16, 33, 3, bias=False)
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

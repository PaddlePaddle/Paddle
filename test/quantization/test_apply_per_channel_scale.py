# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import struct
import unittest

import numpy as np

import paddle
import paddle.nn.quant as Q
from paddle.base import core


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


class ApplyPerChannelScaleTest(unittest.TestCase):
    def config(self):
        self.rows = 32
        self.cols = 128
        self.rtol = 1e-5
        self.atol = 1e-8
        self.dtype = 'float16'
        self.static = False

    def setUp(self):
        self.config()
        paddle.set_default_dtype(self.dtype)
        self.x = paddle.to_tensor(
            np.random.random(size=(self.rows, self.cols)), self.dtype
        )

        self.scales = paddle.to_tensor(
            np.random.uniform(0, 1, size=(self.cols)), self.dtype
        )
        self.out_expected = paddle.multiply(self.x, self.scales)

    def get_out_static(self):
        paddle.enable_static()
        main = paddle.static.Program()
        start = paddle.static.Program()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data("x", self.x.shape, dtype=self.dtype)
            scales = paddle.static.data(
                "scales", self.scales.shape, dtype=self.dtype
            )
            x.stop_gradient = True
            scales.stop_gradient = True
            out = Q.apply_per_channel_scale(x, scales)
            feed_dict = {
                'x': self.x.numpy(),
                'scales': self.scales.numpy(),
            }

            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        paddle.disable_static()
        return out

    def test_apply_per_channel_scale(self):
        if self.static:
            self.out_real = self.get_out_static()
        else:
            paddle.disable_static()
            self.out_real = Q.apply_per_channel_scale(
                x=self.x,
                scales=self.scales,
            )
        out_expected = self.out_expected
        if self.dtype == 'bfloat16' and isinstance(
            self.out_real, paddle.Tensor
        ):
            self.out_real = convert_uint16_to_float(self.out_real)
            out_expected = convert_uint16_to_float(self.out_expected)

        np.testing.assert_allclose(
            out_expected, self.out_real, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class ApplyPerChannelScaleTestCase1(ApplyPerChannelScaleTest):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'


class ApplyPerChannelScaleTestCase2(ApplyPerChannelScaleTest):
    def config(self):
        super().config()
        self.rows = 1024
        self.cols = 128


class ApplyPerChannelScaleStaticTest(ApplyPerChannelScaleTest):
    def config(self):
        super().config()
        self.static = True


if __name__ == '__main__':
    unittest.main()

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


class PreQuantScaleTest(unittest.TestCase):
    def config(self):
        self.rows = 32
        self.cols = 128
        self.rtol = 1e-5
        self.atol = 1e-2
        self.dtype = 'float16'

    def setUp(self) -> None:
        self.config()

        self.x = np.random.random(size=(self.rows, self.cols)).astype(
            self.dtype
        )

        self.scales = np.random.uniform(0, 1, size=(self.cols))

        self.out_expected = np.multiply(self.x, self.scales)

    def test_pre_quant_scale(self):
        self.out_real = Q.apply_per_channel_scale(
            x=paddle.to_tensor(self.x, self.dtype),
            scales=paddle.to_tensor(self.scales, self.dtype),
        )
        print(f"out_real: {self.out_real}, out_expected: {self.out_expected}")
        np.testing.assert_allclose(
            self.out_expected, self.out_real, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()

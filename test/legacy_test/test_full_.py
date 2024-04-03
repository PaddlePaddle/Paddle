#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import struct
import unittest

import numpy as np

import paddle
from paddle import _C_ops
from paddle.base import core


def convert_float_to_uint16(x):
    output = struct.unpack('<I', struct.pack('<f', x))[0] >> 16
    output = np.uint16(output)

    return output


class TestFP16Full_(unittest.TestCase):
    def setUp(self):
        self.type = 'float16'
        self.shape = [30, 10, 2]
        self.value = 1.1

    def test_cuda_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CUDAPlace(0)
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )

    def test_cpu_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CPUPlace()
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )


class TestFP32Full_(unittest.TestCase):
    def setUp(self):
        self.type = 'float32'
        self.shape = [30, 10, 2]
        self.value = 1.1

    def test_cuda_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CUDAPlace(0)
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )

    def test_cpu_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CPUPlace()
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )


class TestFP64Full_(unittest.TestCase):
    def setUp(self):
        self.type = 'float64'
        self.shape = [30, 10, 2]
        self.value = 1.1

    def test_cuda_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CUDAPlace(0)
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )

    def test_cpu_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CPUPlace()
        )

        assert np.array_equal(
            data, np.full(self.shape, self.value, dtype=self.type)
        )


class TestBF16Full_(unittest.TestCase):
    def setUp(self):
        self.type = 'bfloat16'
        self.shape = [10, 2]
        self.value = 1.1

    def test_cuda_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(
            data, data.shape, float(self.value), data.dtype, core.CUDAPlace(0)
        )

        assert np.array_equal(
            data,
            np.full(
                self.shape, convert_float_to_uint16(self.value), dtype='uint16'
            ),
        )

    def test_cpu_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        _C_ops.full_(data, data.shape, self.value, data.dtype, core.CPUPlace())

        assert np.array_equal(
            data,
            np.full(
                self.shape, convert_float_to_uint16(self.value), dtype='uint16'
            ),
        )


if __name__ == "__main__":
    unittest.main()

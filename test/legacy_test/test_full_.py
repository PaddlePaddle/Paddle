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

import unittest

import numpy as np

import paddle
from paddle import _C_ops
from paddle.base import core


def test_api_with_place(data, np_data, value, place):
    _C_ops.full_(data, data.shape, float(value), data.dtype, place)
    np.testing.assert_array_equal(data, np_data)


class TestFull_(unittest.TestCase):
    def setUp(self):
        self.type = 'float32'
        self.shape = [30, 10, 2]
        self.value = 1.1
        self.with_gpu = True if paddle.device.is_compiled_with_cuda() else False

    def test_api(self):
        data = paddle.rand(self.shape, dtype=self.type)
        np_data = np.full(self.shape, self.value, dtype=self.type)
        test_api_with_place(data, np_data, self.value, core.CPUPlace())
        if self.with_gpu:
            test_api_with_place(data, np_data, self.value, core.CUDAPlace(0))


class TestFP16Full_(TestFull_):
    def setUp(self):
        self.type = 'float16'
        self.shape = [30, 10, 2]
        self.value = 1.1
        self.with_gpu = True if paddle.device.is_compiled_with_cuda() else False


class TestFP64Full_(TestFull_):
    def setUp(self):
        self.type = 'float64'
        self.shape = [30, 10, 2]
        self.value = 1.1
        self.with_gpu = True if paddle.device.is_compiled_with_cuda() else False


if __name__ == "__main__":
    unittest.main()

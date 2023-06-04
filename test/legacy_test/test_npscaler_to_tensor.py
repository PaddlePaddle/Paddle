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

DTYPE_MAP = {
    paddle.bool: np.bool_,
    paddle.int32: np.int32,
    paddle.int64: np.int64,
    paddle.float16: np.float16,
    paddle.float32: np.float32,
    paddle.float64: np.float64,
    paddle.complex64: np.complex64,
}


class NumpyScaler2Tensor(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.x_np = np.array([1], dtype=self.dtype)[0]

    def test_dynamic_scaler2tensor(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        self.assertEqual(DTYPE_MAP[x.dtype], self.dtype)
        self.assertEqual(x.numpy(), self.x_np)
        if self.dtype in [
            np.bool_
        ]:  # bool is not supported convert to 0D-Tensor
            return
        self.assertEqual(len(x.shape), 0)

    def test_static_scaler2tensor(self):
        if self.dtype in [np.float16, np.complex64]:
            return
        paddle.enable_static()
        x = paddle.to_tensor(self.x_np)
        self.assertEqual(DTYPE_MAP[x.dtype], self.dtype)
        if self.dtype in [
            np.bool_,
            np.float64,
        ]:  # bool is not supported convert to 0D-Tensor and float64 not supported in static mode
            return
        self.assertEqual(len(x.shape), 0)


class NumpyScaler2TensorBool(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.bool_
        self.x_np = np.array([1], dtype=self.dtype)[0]


class NumpyScaler2TensorFloat16(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.float16
        self.x_np = np.array([1], dtype=self.dtype)[0]


class NumpyScaler2TensorFloat64(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.float64
        self.x_np = np.array([1], dtype=self.dtype)[0]


class NumpyScaler2TensorInt32(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.int32
        self.x_np = np.array([1], dtype=self.dtype)[0]


class NumpyScaler2TensorInt64(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.int64
        self.x_np = np.array([1], dtype=self.dtype)[0]


class NumpyScaler2TensorComplex64(NumpyScaler2Tensor):
    def setUp(self):
        self.dtype = np.complex64
        self.x_np = np.array([1], dtype=self.dtype)[0]

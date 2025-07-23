# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid
from test_trapezoid import (
    Testfp16Trapezoid,
    TestTrapezoidAPI,
    TestTrapezoidError,
)

import paddle


class TestCumulativeTrapezoidAPI(TestTrapezoidAPI):
    def set_api(self):
        self.ref_api = cumulative_trapezoid
        self.paddle_api = paddle.cumulative_trapezoid


class TestCumulativeTrapezoidWithX(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float32')
        self.dx = None
        self.axis = -1


class TestCumulativeTrapezoidAxis(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = 0


class TestCumulativeTrapezoidWithDx(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 3.0
        self.axis = -1


class TestCumulativeTrapezoidfloat64(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float64')
        self.dx = None
        self.axis = -1


class TestCumulativeTrapezoidWithOutDxX(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = None
        self.dx = None
        self.axis = -1


class TestCumulativeTrapezoidBroadcast(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = np.random.random(3).astype('float32')
        self.dx = None
        self.axis = 1


class TestCumulativeTrapezoidAxis1(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = None
        self.dx = 1
        self.axis = 1


class TestCumulativeTrapezoidError(TestTrapezoidError):
    def set_api(self):
        self.paddle_api = paddle.cumulative_trapezoid


class Testfp16CumulativeTrapezoid(Testfp16Trapezoid):
    def set_api(self):
        self.paddle_api = paddle.cumulative_trapezoid
        self.ref_api = cumulative_trapezoid


class TestCumulativeTrapezoidZeroSizeTensorCase1(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 0)).astype('float32')
        self.x = np.random.random(3).astype('float32')
        self.dx = None
        self.axis = 0


class TestCumulativeTrapezoidZeroSizeTensorCase2(TestCumulativeTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((1, 3, 3)).astype('float32')
        self.x = np.random.random((0, 3, 3)).astype('float32')
        self.dx = None
        self.axis = -1


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

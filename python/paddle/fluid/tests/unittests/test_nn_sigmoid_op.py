# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from op_test import OpTest
from scipy.special import expit, erf
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as functional


class TestNNSigmoidAPI(unittest.TestCase):

    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def ref_backward(self, y, dy):
        return dy * y * (1 - y)

    def check_static_api(self, place):
        paddle.enable_static()
        main_program = paddle.static.Program()
        mysigmoid = nn.Sigmoid(name="api_sigmoid")
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=self.x_shape)
            x.stop_gradient = False
            y = mysigmoid(x)
            fluid.backward.append_backward(paddle.mean(y))
        exe = paddle.static.Executor(place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], self.y, rtol=1e-05)
        self.assertTrue(y.name.startswith("api_sigmoid"))

    def check_dynamic_api(self, place):
        paddle.disable_static(place)
        x = paddle.to_tensor(self.x)
        mysigmoid = nn.Sigmoid()
        y = mysigmoid(x)
        np.testing.assert_allclose(y.numpy(), self.y, rtol=1e-05)

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.check_dynamic_api(place)
            self.check_static_api(place)


class TestNNFunctionalSigmoidAPI(unittest.TestCase):

    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def check_static_api(self, place):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = functional.sigmoid(x, name="api_sigmoid")
        exe = paddle.static.Executor(fluid.CPUPlace())
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], self.y, rtol=1e-05)

    def check_dynamic_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = functional.sigmoid(x)
        np.testing.assert_allclose(y.numpy(), self.y, rtol=1e-05)

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.check_static_api(place)
            self.check_dynamic_api()

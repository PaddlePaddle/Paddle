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
import sys
import unittest
from os.path import dirname

import numpy as np

import paddle

sys.path.append(dirname(dirname(__file__)))
import utils


class ReduceSumSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis):
        return paddle.sum(x, axis=axis, keepdim=True)


class TestReduceSumBase(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()
        self.prepare_atol()
        self.x = paddle.randn(self.shape, dtype="float32")

    def prepare_data(self):
        self.shape = [8, 4]
        self.axis = [-1]

    def prepare_atol(self):
        self.atol = 1e-5

    def eval(self, use_cinn):
        net = ReduceSumSubGraph()
        net.eval()
        net = utils.apply_to_static(net, use_cinn)
        out = net(self.x, self.axis)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)

        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=self.atol
        )


class TestReduceLastAxis1(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 64]
        self.axis = -1


class TestReduceLastAxis2(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 1024]
        self.axis = -1

    def prepare_atol(self):
        self.atol = 2e-5


class TestReduceLastDim3(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 2048]
        self.axis = -1

    def prepare_atol(self):
        self.atol = 1e-4


class TestReduceFirstAxis1(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 8]
        self.axis = 0


class TestReduceFirstAxis2(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 64]
        self.axis = 0


class TestReduceFirstAxis3(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [512, 128]
        self.axis = 0


class TestReduceFirstAxis4(TestReduceSumBase):
    def prepare_data(self):
        self.shape = [4, 8]
        self.axis = 0


if __name__ == '__main__':
    unittest.main()

#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle
from paddle import nn

np.random.seed(1)


def apply_to_static(support_to_static, model, image_shape=None):
    if support_to_static:
        specs = None
        model = paddle.jit.to_static(model, input_spec=specs)

    return model


class Layer0(nn.Layer):
    def __init__(self, level):
        super().__init__()
        self._linear1 = nn.Linear(10, 5)
        self._linear2 = nn.Linear(10, 5)
        self.layer1 = Layer1(level)
        self.layer1 = apply_to_static(True, self.layer1)

    def forward(self, x):
        out1 = self._linear1(x)
        out2 = self._linear2(x)
        # out2.stop_gradient = True not raise error
        a = [out1, out2]
        b = self.layer1(a)
        # self.layer1(out1, out2) will raise error
        return b


class Layer1(nn.Layer):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self._linear = nn.Linear(5, 2)

    def forward(self, x):
        inp = x[self.level]
        val = self._linear(inp)
        return val


class TestDuplicateOutput(Dy2StTestBase):
    def test_case(self):
        # create network
        layer = Layer0(0)
        a = paddle.rand(shape=[10, 10])
        out = layer(a)
        loss = out.mean()
        loss.backward()


if __name__ == '__main__':
    unittest.main()

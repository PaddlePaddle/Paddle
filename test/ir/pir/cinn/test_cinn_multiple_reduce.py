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
import unittest

import numpy as np
import utils

import paddle
from paddle.base import core


class MultipleReduceSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x.sum(axis=-1)
        x2 = (x * x).sum(axis=-1)
        return x1 + x2


class TestMultipleReduceSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [32, 32]
        self.dtype = "float32"
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn(self.shape, dtype=self.dtype)
        self.x.stop_gradient = False

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = MultipleReduceSubGraph()
        net = utils.apply_to_static(net, use_cinn=use_cinn)
        net.eval()
        out = net(self.x)

        core._set_prim_all_enabled(False)
        return out

    def test_cinn(self):
        cinn_out = self.eval(use_cinn=True, use_prim=True)
        dy_out = self.eval(use_cinn=False, use_prim=True)
        np.testing.assert_allclose(cinn_out, dy_out, atol=1e-2)


if __name__ == '__main__':
    unittest.main()

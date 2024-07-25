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


class GroupNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x1 = paddle.tensor.manipulation.concat([x, y], axis=0)
        return x1 + x1


class TestGroupNormSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.shape = [1, 33, 256, 256]
        self.dtype = "float32"
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.rand(self.shape, dtype=self.dtype)
        self.y = paddle.rand(self.shape, dtype=self.dtype)
        self.x.stop_gradient = False

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = GroupNormSubGraph()
        net = utils.apply_to_static(net, use_cinn=use_cinn)
        net.eval()
        out = net(self.x, self.y)

        core._set_prim_all_enabled(False)
        return out

    def test_cinn(self):
        dy_out = self.eval(use_cinn=False, use_prim=True)
        cinn_out = self.eval(use_cinn=True, use_prim=True)
        np.testing.assert_allclose(cinn_out, dy_out, atol=1e-50)


if __name__ == '__main__':
    unittest.main()

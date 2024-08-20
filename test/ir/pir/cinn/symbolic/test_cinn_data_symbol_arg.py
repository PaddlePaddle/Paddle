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
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

import unittest

import numpy as np
import utils

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [64, 512, 8, 8], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [4], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
    ):
        var_4 = var_1.__getitem__(0)
        var_5 = var_1.__getitem__(1)
        var_6 = var_0.reshape([var_4, 8, 8, var_5, var_2, var_3])
        var_7 = var_6.transpose([0, 4, 5, 3, 1, 2])
        var_8 = var_7.reshape([-1, 512, 8, 8])
        return var_8


def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(
            shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False
        ),
        paddle.static.InputSpec(
            shape=(4,), dtype=paddle.int32, stop_gradient=False
        ),
        paddle.static.InputSpec(
            shape=(1,), dtype=paddle.int32, stop_gradient=False
        ),
        paddle.static.InputSpec(
            shape=(1,), dtype=paddle.int32, stop_gradient=False
        ),
    )
    return inputspec


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[64, 512, 8, 8], dtype=paddle.float32),
        paddle.to_tensor([4, 128, 1, 1], dtype=paddle.int32),
        paddle.to_tensor(4, dtype=paddle.int32),
        paddle.to_tensor(16, dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[64, 512, 8, 8]).astype("float32"),
        np.array([4, 128, 1, 1], dtype="int32"),
        np.array(4, dtype="int32"),
        np.array(16, dtype="int32"),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()

    def train(self, net, use_cinn=False):
        net = utils.apply_to_static(self.net, use_cinn, create_inputspec())
        net.eval()
        paddle.seed(123)
        out = net(*self.inputs)
        return out

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, use_cinn=False)
        cinn_out = self.train(self.net, use_cinn=True)
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base


class MLP(paddle.nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        self._linear1 = paddle.nn.Linear(
            input_size,
            3,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.1)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.1)
            ),
        )
        self._linear2 = paddle.nn.Linear(
            3,
            4,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.1)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.1)
            ),
        )

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = self._linear2(x)
        x = paddle.sum(x)
        return x


class TestDygraphFramework(unittest.TestCase):
    def test_dygraph_to_string(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with base.dygraph.guard():
            var_inp = paddle.to_tensor(np_inp)
            print(str(var_inp))


if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()

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
from op_test import OpTest

import paddle

np.random.seed(100)
paddle.seed(100)


def sum_as_net(x, y):
    out_dtype = x.dtype
    diff = len(x.shape) - len(y.shape)
    axis = []
    for i in range(diff):
        axis.append(i)
    for i in len(y.shape):
        if y.shape[i] != x.shape[i + diff]:
            axis.append(i + diff)
    return np.sum(x, axis=axis, dtype=out_dtype)


class TestSumAsOp(OpTest):
    def setUp(self):
        self.op_type = "sum_as"
        self.python_api = paddle.sum_as
        self.init_config()
        self.inputs = {'x': self.x, 'y': self.y}
        self.target = sum_as_net(self.inputs['x'], self.inputs['y'])
        self.outputs = {'out': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # self.check_grad(['x', 'y'], ['out'])
        pass

    def init_config(self):
        self.x = np.random.randn(200, 600).astype('float64')
        self.y = np.random.randn(200, 600).astype('float64')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

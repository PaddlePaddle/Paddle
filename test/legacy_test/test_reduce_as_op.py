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


def reduce_as_net(x, target):
    return paddle.reduce_as(x, target)


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestSumAsOp(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        self.init_input()
        self.init_attrs()
        self.calc_output()

        self.python_api = paddle.reduce_as
        self.op_type = "reduce_as"
        self.inputs = {'x': self.x, 'target': self.y}
        self.outputs = {'out': self.out}
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape_x = [10, 10, 6]
        self.shape_y = [10, 6]

    def init_input(self):
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.y = np.random.random(self.shape_y).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def if_enable_cinn(self):
        pass

    def calc_output(self):
        self.out = self.x.sum(axis=tuple(self.attrs['dim']))

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

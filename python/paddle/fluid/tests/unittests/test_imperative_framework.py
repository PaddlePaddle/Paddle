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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
import numpy as np
from test_imperative_base import new_program_scope
from paddle.fluid.framework import _test_eager_guard


class MLP(fluid.Layer):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self._linear1 = fluid.dygraph.Linear(
            input_size,
            3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)))
        self._linear2 = fluid.dygraph.Linear(
            3,
            4,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = self._linear2(x)
        x = fluid.layers.reduce_sum(x)
        return x


class TestDygraphFramework(unittest.TestCase):
    def func_test_dygraph_backward(self):
        with new_program_scope():
            mlp = MLP(input_size=2)
            var_inp = fluid.layers.data(
                "input", shape=[2, 2], dtype="float32", append_batch_size=False)
            out = mlp(var_inp)
            try:
                out.backward()
                raise AssertionError(
                    "backward should not be usable in static graph mode")
            except AssertionError as e:
                self.assertTrue((e is not None))

    def test_dygraph_backward(self):
        with _test_eager_guard():
            self.func_test_dygraph_backward()
        self.func_test_dygraph_backward()

    def func_test_dygraph_to_string(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with fluid.dygraph.guard():
            var_inp = fluid.dygraph.to_variable(np_inp)
            print(str(var_inp))

    def test_dygraph_to_string(self):
        with _test_eager_guard():
            self.func_test_dygraph_to_string()
        self.func_test_dygraph_to_string()

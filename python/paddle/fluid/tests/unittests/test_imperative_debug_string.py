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


class TestDygraphDebugString(unittest.TestCase):
    def test_dygraph_debug_string(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        unique_name = 0
        trace_var = 0
        alive_var = 0
        with fluid.dygraph.guard():
            mlp = MLP(input_size=2)
            for i in range(10):
                var_inp = fluid.dygraph.base.to_variable(np_inp)
                out = mlp(var_inp)
                out.backward()
                mlp.clear_gradients()
                unique_name_tmp, trace_var_tmp, alive_var_tmp = fluid.dygraph.base._print_debug_msg(
                    mlp.parameters(), is_test=True)
                if i > 0:
                    self.assertGreaterEqual(unique_name, unique_name_tmp)
                    self.assertGreaterEqual(trace_var, trace_var_tmp)
                    self.assertGreaterEqual(alive_var, alive_var_tmp)
                else:
                    unique_name = unique_name_tmp
                    trace_var = trace_var_tmp
                    alive_var = alive_var_tmp
                try:
                    fluid.dygraph.base._print_debug_msg(mlp.parameters())
                except Exception as e:
                    raise RuntimeError(
                        "No Exception is accepted in _print_debug_msg, but we got: {}".
                        format(e))

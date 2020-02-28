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

from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.jit import dygraph_to_static_graph

np.random.seed(1)

if fluid.is_compiled_with_cuda():
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()


def dyfunc_with_if_else(x_v, label=None):
    if fluid.layers.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    # plain if in python
    if label is not None:
        loss = fluid.layers.cross_entropy(x_v, label)
        return loss
    return x_v


def dyfunc_with_if_else2(x, col=100):
    row = 0
    # plain if in python
    if abs(col) > x.shape[-1]:
        col = -1
    if fluid.layers.reduce_mean(x).numpy()[0] > x.numpy()[row][col]:
        y = fluid.layers.relu(x)
    else:
        x_pow = fluid.layers.pow(x, 2)
        y = fluid.layers.tanh(x_pow)
    return y


def nested_if_else(x_v):
    batch_size = 16
    feat_size = x_v.shape[-1]
    bias = fluid.layers.fill_constant([feat_size], dtype='float32', value=1)
    # plain if in python
    if x_v.shape[0] != batch_size:
        batch_size = x_v.shape[0]
    if fluid.layers.mean(x_v).numpy()[0] < 0:
        y = x_v + bias
        w = fluid.layers.fill_constant([feat_size], dtype='float32', value=10)
        if y.numpy()[0] < 10:
            tmp = y * w
            y = fluid.layers.relu(tmp)
            if fluid.layers.mean(y).numpy()[0] < batch_size:
                y = fluid.layers.abs(y)
            else:
                tmp = fluid.layers.fill_constant(
                    [feat_size], dtype='float32', value=-1)
                y = y - tmp
    else:
        y = x_v - bias
    return y


class TestDygraphIfElse(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_v = fluid.layers.assign(self.x)
            # Transform into static graph
            out = dygraph_to_static_graph(self.dyfunc)(x_v)
            exe = fluid.Executor(place)
            ret = exe.run(main_program, fetch_list=out)
            return ret

    def _run_dygraph(self):
        with fluid.dygraph.guard(place):
            x_v = fluid.dygraph.to_variable(self.x)
            ret = self.dyfunc(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2


class TestDygraphIfElse3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else


if __name__ == '__main__':
    unittest.main()

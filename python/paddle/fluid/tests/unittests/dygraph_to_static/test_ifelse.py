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
import unittest

from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator

from ifelse_simple_func import *

np.random.seed(1)

if fluid.is_compiled_with_cuda():
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()


class TestDygraphIfElse(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):

        with fluid.dygraph.guard(place):
            x_v = fluid.dygraph.to_variable(self.x)
            if to_static:
                ret = declarative(self.dyfunc)(x_v)
            else:
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


class TestDygraphIfElse4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_2


class TestDygraphIfElse5(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_3


def dyfunc_ifExp_with_while(x):
    y = [x]

    def add_fn(x):
        x = x + 1
        return x

    def cond(i, ten, y):
        return i < ten

    def map_func(func, tensor_list):
        return [func(x) for x in tensor_list]

    def body(i, ten, y):
        # It will be converted into `layers.cond` as followed.
        # map_func(lambda x: fluid.layers.cond(i==0, lambda: x, lambda: add_fn(x), y)
        y = map_func(lambda x: x if (i == 0) is not None else add_fn(x), y)
        i += 1
        return [i, ten, y]

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    ten = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    i, ten, y = fluid.layers.while_loop(cond, body, [i, ten, y])
    return y[0]


class TestDygraphIfElse6(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp_with_while


def dyfunc_ifExp(x):
    y = [x]

    def add_fn(x):
        x = x + 1
        return x

    def map_func(func, tensor_list):
        return [func(x) for x in tensor_list]

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    # It will be converted into `layers.cond` as followed.
    # map_func(lambda x: fluid.layers.cond(i==1, lambda: x, lambda: add_fn(x), y)
    # `if (Tensor) == 1` is supported in dygraph.
    y = map_func(lambda x: x if i == 1 else add_fn(x), y)
    return y[0]


class TestDygraphIfElse7(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp


class TestDygraphIfElseWithAndOr(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or


class TestDygraphIfElseWithAndOr1(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_1


class TestDygraphIfElseWithAndOr2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_2


class TestDygraphIfElseWithAndOr3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_3


class TestDygraphIfElseWithAndOr4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_4


class TestDygraphIfElseWithClassVar(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_class_var


class TestDygraphIfTensor(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_tensor_case


class TestDygraphIfElseNet(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithControlFlowIf

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static=False):
        prog_trans = ProgramTranslator()
        prog_trans.enable(to_static)

        with fluid.dygraph.guard(place):
            net = self.Net()
            x_v = fluid.dygraph.to_variable(self.x)
            ret = net(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


# Test to call function ahead caller.
def relu(x):
    return fluid.layers.relu(x)


def call_external_func(x, label=None):
    if fluid.layers.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = add_fn(x)

    x_v = relu(x_v)
    if label is not None:
        loss = loss_fn(x_v, label)
        return loss
    return x_v


class TestAst2FuncWithExternalFunc(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = call_external_func


class NetWithExternalFunc(fluid.dygraph.Layer):
    @declarative
    def forward(self, x, label=None):
        if fluid.layers.mean(x) < 0:
            x_v = x - 1
        else:
            x_v = add_fn(x)

        x_v = softmax(x_v)
        if label is not None:
            loss = loss_fn(x_v, label)
            return loss
        return x_v


# Test to call function behind caller.
def softmax(x):
    return fluid.layers.softmax(x)


class TestNetWithExternalFunc(TestDygraphIfElseNet):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithExternalFunc


if __name__ == '__main__':
    unittest.main()

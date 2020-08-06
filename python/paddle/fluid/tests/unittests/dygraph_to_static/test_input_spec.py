# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, declarative, ProgramTranslator, Layer, jit
from paddle.fluid.dygraph import TensorSpec

import unittest

program_trans = ProgramTranslator()


class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = fluid.dygraph.Linear(10, 3)

    @declarative(
        input_signature=[TensorSpec(
            shape=[None, 10], dtype='float32')])
    def forward(self, x, a=1, b=2):
        y = self.inner_function(x)
        return y

    @declarative
    def inner_function(self, x):
        y = self.linear(x)
        return y

    @declarative
    def ff(self, x, y):
        z = x + y
        return z


class TestInputSpec(unittest.TestCase):
    def setUp(self):
        pass

    def test_input_spec(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))

            net = SimpleNet()
            y = net(x, a=1)

            # TODO: add check for input with input_spec
            x_2 = to_variable(np.ones([4, 20]).astype('float32'))
            z = net.ff(x_2, np.ones([20]).astype('float32'))

            # kwargs and input_signature should not be specificed in same time
            with self.assertRaises(ValueError):
                net(x, a=1, other_kwarg=2)

            # if specific input_signature, we will check whether the input argument is legal.
            # with self.assertRaises(ValueError):
            #     y = net(x_2)


@declarative
def foo(a, b, c=1, d=2):
    z = a + b
    return z


class TestDifferentInputSpecCacheProgram(unittest.TestCase):
    def test_with_different_input(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x_data = np.ones([16, 10]).astype('float32')
            y_data = np.ones([10]).astype('float32') * 2
            z_data = np.ones([10]).astype('float32') * 2.2

            # [16, 10] + [10] (varbase)
            out_1 = foo(to_variable(x_data), to_variable(y_data))
            self.assertTrue(np.allclose(x_data + y_data, out_1.numpy()))
            self.assertTrue(len(foo.program_cache) == 1)

            # [16, 10] + [10] (numpy)
            out_2 = foo(to_variable(x_data), y_data)
            self.assertTrue(np.allclose(x_data + y_data, out_2.numpy()))
            self.assertTrue(len(foo.program_cache) == 1)

            # [16, 10] + [10] (numpy)
            out_3 = foo(to_variable(x_data), z_data)
            self.assertTrue(np.allclose(x_data + z_data, out_3.numpy()))
            # hit cache program
            self.assertTrue(len(foo.program_cache) == 1)

            # [16, 10] + [10] (numpy) with other different arguments (c=3)
            out_4 = foo(to_variable(x_data), z_data, 3)
            self.assertTrue(np.allclose(x_data + z_data, out_4.numpy()))
            # create a new program
            self.assertTrue(len(foo.program_cache) == 2)


if __name__ == '__main__':
    unittest.main()

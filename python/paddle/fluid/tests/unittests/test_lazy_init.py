# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
import numpy as np
from paddle import LazyInit
from paddle.nn import Linear
from paddle.nn.initializer import *
from paddle.fluid import unique_name


class TestInitializerBase(unittest.TestCase):

    def setUp(self):
        self.set_initializer()
        self.set_param_attr()
        self.set_init_ops()
        self.clear_nameset()

    def set_initializer(self):
        self.w_initializer = Constant(0.6)
        self.b_initializer = Constant(0.3)

    def set_param_attr(self):
        self.weight_attr = paddle.ParamAttr(name="weight",
                                            initializer=self.w_initializer)

        self.bias_attr = paddle.ParamAttr(name="bias",
                                          initializer=self.b_initializer)

    def set_init_ops(self):
        self.init_ops = ['fill_constant', 'fill_constant']

    def clear_nameset(self):
        unique_name.dygraph_parameter_name_checker._name_set = set()

    def test_wrapper(self):
        fc = LazyInit(Linear)(10,
                              10,
                              weight_attr=self.weight_attr,
                              bias_attr=self.bias_attr)
        program = fc.startup_program
        self.check_program(program)

    def check_program(self, program):
        self.assertEqual(program.block(0).var("weight").shape, (10, 10))
        self.assertEqual(program.block(0).var("bias").shape, (10, ))
        ops = [op.type for op in program.block(0).ops]
        self.assertEqual(ops, self.init_ops)


class TestDygraphLazy(TestInitializerBase):

    def test_wrapper(self):
        fc = LazyInit(Linear)(10,
                              10,
                              weight_attr=self.weight_attr,
                              bias_attr=self.bias_attr)

        self.check_data(fc)

    def check_data(self, model):
        x = paddle.randn([2, 10])
        # weight and bias have no memory
        with self.assertRaises(RuntimeError):
            out = model(x)

        for param in model.parameters():
            param.initialize()

        out = model(x)
        self.assertEqual(out.shape, [2, 10])

        self.assertTrue(
            np.array_equal(model.weight.numpy(),
                           np.ones([10, 10], dtype=np.float32) * 0.6))
        self.assertTrue(
            np.array_equal(model.bias.numpy(),
                           np.ones([10], dtype=np.float32) * 0.3))


class TestUniform(TestInitializerBase):

    def set_initializer(self):
        self.w_initializer = Uniform()
        self.b_initializer = Uniform()

    def set_init_ops(self):
        self.init_ops = ['uniform_random', 'uniform_random']


class TestNormal(TestInitializerBase):

    def set_initializer(self):
        self.w_initializer = Normal()
        self.b_initializer = Normal()

    def set_init_ops(self):
        self.init_ops = ['gaussian_random', 'gaussian_random']


class TestTruncatedNormal(TestInitializerBase):

    def set_initializer(self):
        self.w_initializer = TruncatedNormal()
        self.b_initializer = TruncatedNormal()

    def set_init_ops(self):
        self.init_ops = [
            'truncated_gaussian_random', 'truncated_gaussian_random'
        ]


class TestXavierNormal(TestNormal):

    def set_initializer(self):
        self.w_initializer = XavierNormal()
        self.b_initializer = XavierNormal()


class TestXavierUniform(TestUniform):

    def set_initializer(self):
        self.w_initializer = XavierUniform()
        self.b_initializer = XavierUniform()


if __name__ == '__main__':
    unittest.main()

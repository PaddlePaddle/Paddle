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

import copy
import unittest

import numpy as np

import paddle
from paddle import LazyGuard
from paddle.base import unique_name
from paddle.nn import Layer, Linear
from paddle.nn.initializer import (
    Constant,
    Normal,
    TruncatedNormal,
    Uniform,
    XavierNormal,
    XavierUniform,
)


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
        self.weight_attr = paddle.ParamAttr(
            name="weight", initializer=self.w_initializer
        )

        self.bias_attr = paddle.ParamAttr(
            name="bias", initializer=self.b_initializer
        )

    def set_init_ops(self):
        self.init_ops = ['fill_constant', 'fill_constant']

    def clear_nameset(self):
        unique_name.dygraph_parameter_name_checker._name_set = set()

    def test_wrapper(self):
        paddle.disable_static()
        with LazyGuard():
            fc = Linear(
                10,
                10,
                weight_attr=self.weight_attr,
                bias_attr=self.bias_attr,
            )
        program = fc._startup_program()
        if not paddle.framework.use_pir_api():
            self.check_program(program)

    def check_program(self, program):
        self.assertEqual(program.block(0).var("weight").shape, (10, 10))
        self.assertEqual(program.block(0).var("bias").shape, (10,))
        ops = [op.type for op in program.block(0).ops]
        self.assertEqual(ops, self.init_ops)


class TestDygraphLazy(TestInitializerBase):
    def test_wrapper(self):
        with LazyGuard():
            fc = Linear(
                10, 10, weight_attr=self.weight_attr, bias_attr=self.bias_attr
            )

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

        np.testing.assert_allclose(
            model.weight.numpy(), np.ones([10, 10], dtype=np.float32) * 0.6
        )
        np.testing.assert_allclose(
            model.bias.numpy(), np.ones([10], dtype=np.float32) * 0.3
        )


class NestModel(Layer):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc = Linear(10, 10)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x


class TestNestModelLazy(TestInitializerBase):
    def test_wrapper(self):
        paddle.disable_static()
        with LazyGuard():
            base_model = Linear(
                10,
                10,
                weight_attr=self.weight_attr,
                bias_attr=self.bias_attr,
            )
            nest_model = NestModel(base_model)

        self.check_data(nest_model)
        if not paddle.framework.use_pir_api():
            self.check_program(nest_model)

    def check_data(self, model):
        x = paddle.randn([2, 10])
        # weight and bias have no memory
        with self.assertRaises(RuntimeError):
            out = model(x)

        for param in model.parameters():
            param.initialize()

        out = model(x)
        self.assertEqual(out.shape, [2, 10])

        np.testing.assert_allclose(
            model.base_model.weight.numpy(),
            np.ones([10, 10], dtype=np.float32) * 0.6,
        )
        np.testing.assert_allclose(
            model.base_model.bias.numpy(), np.ones([10], dtype=np.float32) * 0.3
        )

    def check_program(self, model):
        # verify nest_model startup_program
        whole_program = model._startup_program()
        self.assertEqual(whole_program.block(0).var("weight").shape, (10, 10))
        self.assertEqual(whole_program.block(0).var("bias").shape, (10,))
        ops = [op.type for op in whole_program.block(0).ops]
        init_ops = [*self.init_ops, 'uniform_random', 'fill_constant']
        self.assertEqual(ops, init_ops)

        # verify base_model startup_program
        sub_program = model.base_model._startup_program()
        self.assertEqual(sub_program.block(0).var("weight").shape, (10, 10))
        self.assertEqual(sub_program.block(0).var("bias").shape, (10,))
        ops = [op.type for op in sub_program.block(0).ops]
        self.assertEqual(ops, self.init_ops)


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
            'truncated_gaussian_random',
            'truncated_gaussian_random',
        ]


class TestXavierNormal(TestNormal):
    def set_initializer(self):
        self.w_initializer = XavierNormal()
        self.b_initializer = XavierNormal()


class TestXavierUniform(TestUniform):
    def set_initializer(self):
        self.w_initializer = XavierUniform()
        self.b_initializer = XavierUniform()


class LinearNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 10)


class TestDeepCopyLazyInitializedParam(unittest.TestCase):
    def test_deepcopy_lazy_initialized_param(self):
        paddle.disable_static()
        with LazyGuard():
            net = LinearNet()
            copy.deepcopy(net)


if __name__ == '__main__':
    unittest.main()

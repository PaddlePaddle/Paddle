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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
from paddle import LazyGuard
from paddle.fluid import unique_name
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
=======
import paddle
import unittest
import numpy as np
from paddle import LazyGuard
from paddle.nn import Linear, Layer
from paddle.nn.initializer import *
from paddle.fluid import unique_name


class TestInitializerBase(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_initializer()
        self.set_param_attr()
        self.set_init_ops()
        self.clear_nameset()

    def set_initializer(self):
        self.w_initializer = Constant(0.6)
        self.b_initializer = Constant(0.3)

    def set_param_attr(self):
<<<<<<< HEAD
        self.weight_attr = paddle.ParamAttr(
            name="weight", initializer=self.w_initializer
        )

        self.bias_attr = paddle.ParamAttr(
            name="bias", initializer=self.b_initializer
        )
=======
        self.weight_attr = paddle.ParamAttr(name="weight",
                                            initializer=self.w_initializer)

        self.bias_attr = paddle.ParamAttr(name="bias",
                                          initializer=self.b_initializer)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def set_init_ops(self):
        self.init_ops = ['fill_constant', 'fill_constant']

    def clear_nameset(self):
        unique_name.dygraph_parameter_name_checker._name_set = set()

    def test_wrapper(self):
        with LazyGuard():
<<<<<<< HEAD
            fc = Linear(
                10, 10, weight_attr=self.weight_attr, bias_attr=self.bias_attr
            )
=======
            fc = Linear(10,
                        10,
                        weight_attr=self.weight_attr,
                        bias_attr=self.bias_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        program = fc._startup_program()
        print(program)
        self.check_program(program)

    def check_program(self, program):
        self.assertEqual(program.block(0).var("weight").shape, (10, 10))
<<<<<<< HEAD
        self.assertEqual(program.block(0).var("bias").shape, (10,))
=======
        self.assertEqual(program.block(0).var("bias").shape, (10, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ops = [op.type for op in program.block(0).ops]
        self.assertEqual(ops, self.init_ops)


class TestDygraphLazy(TestInitializerBase):
<<<<<<< HEAD
    def test_wrapper(self):
        with LazyGuard():
            fc = Linear(
                10, 10, weight_attr=self.weight_attr, bias_attr=self.bias_attr
            )
=======

    def test_wrapper(self):
        with LazyGuard():
            fc = Linear(10,
                        10,
                        weight_attr=self.weight_attr,
                        bias_attr=self.bias_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

<<<<<<< HEAD
        np.testing.assert_allclose(
            model.weight.numpy(), np.ones([10, 10], dtype=np.float32) * 0.6
        )
        np.testing.assert_allclose(
            model.bias.numpy(), np.ones([10], dtype=np.float32) * 0.3
        )


class NestModel(Layer):
    def __init__(self, base_model):
        super().__init__()
=======
        np.testing.assert_allclose(model.weight.numpy(),
                                   np.ones([10, 10], dtype=np.float32) * 0.6)
        np.testing.assert_allclose(model.bias.numpy(),
                                   np.ones([10], dtype=np.float32) * 0.3)


class NestModel(Layer):

    def __init__(self, base_model):
        super(NestModel, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.base_model = base_model
        self.fc = Linear(10, 10)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x


class TestNestModelLazy(TestInitializerBase):
<<<<<<< HEAD
    def test_wrapper(self):
        with LazyGuard():
            base_model = Linear(
                10, 10, weight_attr=self.weight_attr, bias_attr=self.bias_attr
            )
=======

    def test_wrapper(self):
        with LazyGuard():
            base_model = Linear(10,
                                10,
                                weight_attr=self.weight_attr,
                                bias_attr=self.bias_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            nest_model = NestModel(base_model)

        self.check_data(nest_model)
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

<<<<<<< HEAD
        np.testing.assert_allclose(
            model.base_model.weight.numpy(),
            np.ones([10, 10], dtype=np.float32) * 0.6,
        )
        np.testing.assert_allclose(
            model.base_model.bias.numpy(), np.ones([10], dtype=np.float32) * 0.3
        )
=======
        np.testing.assert_allclose(model.base_model.weight.numpy(),
                                   np.ones([10, 10], dtype=np.float32) * 0.6)
        np.testing.assert_allclose(model.base_model.bias.numpy(),
                                   np.ones([10], dtype=np.float32) * 0.3)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def check_program(self, model):
        # verify nest_model startup_program
        whole_program = model._startup_program()
        self.assertEqual(whole_program.block(0).var("weight").shape, (10, 10))
<<<<<<< HEAD
        self.assertEqual(whole_program.block(0).var("bias").shape, (10,))
=======
        self.assertEqual(whole_program.block(0).var("bias").shape, (10, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ops = [op.type for op in whole_program.block(0).ops]
        init_ops = self.init_ops + ['uniform_random', 'fill_constant']
        self.assertEqual(ops, init_ops)

        # verify base_model startup_program
        sub_program = model.base_model._startup_program()
        self.assertEqual(sub_program.block(0).var("weight").shape, (10, 10))
<<<<<<< HEAD
        self.assertEqual(sub_program.block(0).var("bias").shape, (10,))
=======
        self.assertEqual(sub_program.block(0).var("bias").shape, (10, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ops = [op.type for op in sub_program.block(0).ops]
        self.assertEqual(ops, self.init_ops)


class TestUniform(TestInitializerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_initializer(self):
        self.w_initializer = Uniform()
        self.b_initializer = Uniform()

    def set_init_ops(self):
        self.init_ops = ['uniform_random', 'uniform_random']


class TestNormal(TestInitializerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_initializer(self):
        self.w_initializer = Normal()
        self.b_initializer = Normal()

    def set_init_ops(self):
        self.init_ops = ['gaussian_random', 'gaussian_random']


class TestTruncatedNormal(TestInitializerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_initializer(self):
        self.w_initializer = TruncatedNormal()
        self.b_initializer = TruncatedNormal()

    def set_init_ops(self):
        self.init_ops = [
<<<<<<< HEAD
            'truncated_gaussian_random',
            'truncated_gaussian_random',
=======
            'truncated_gaussian_random', 'truncated_gaussian_random'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]


class TestXavierNormal(TestNormal):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_initializer(self):
        self.w_initializer = XavierNormal()
        self.b_initializer = XavierNormal()


class TestXavierUniform(TestUniform):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_initializer(self):
        self.w_initializer = XavierUniform()
        self.b_initializer = XavierUniform()


if __name__ == '__main__':
    unittest.main()

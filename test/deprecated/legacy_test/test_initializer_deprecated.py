#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.base import framework

DELTA = 0.00001


class TestSetGlobalInitializer(unittest.TestCase):
    def test_set_global_weight_initializer(self):
        """Test Set Global Param initializer with UniformInitializer"""
        main_prog = framework.Program()
        startup_prog = framework.Program()
        base.set_global_initializer(
            paddle.nn.initializer.Uniform(low=-0.5, high=0.5)
        )
        with base.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[1, 3, 32, 32])
            # default initializer of param in layers.conv2d is NormalInitializer
            conv = paddle.static.nn.conv2d(x, 5, 3)

        block = startup_prog.global_block()
        self.assertEqual(len(block.ops), 2)

        # init weight is the first op, and bias is the second
        bias_init_op = block.ops[1]
        self.assertEqual(bias_init_op.type, 'fill_constant')
        self.assertAlmostEqual(bias_init_op.attr('value'), 0.0, delta=DELTA)

        param_init_op = block.ops[0]
        self.assertEqual(param_init_op.type, 'uniform_random')
        self.assertAlmostEqual(param_init_op.attr('min'), -0.5, delta=DELTA)
        self.assertAlmostEqual(param_init_op.attr('max'), 0.5, delta=DELTA)
        self.assertEqual(param_init_op.attr('seed'), 0)
        base.set_global_initializer(None)

    def test_set_global_bias_initializer(self):
        """Test Set Global Bias initializer with NormalInitializer"""
        main_prog = framework.Program()
        startup_prog = framework.Program()
        base.set_global_initializer(
            paddle.nn.initializer.Uniform(low=-0.5, high=0.5),
            bias_init=paddle.nn.initializer.Normal(0.0, 2.0),
        )
        with base.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[1, 3, 32, 32])
            # default initializer of bias in layers.conv2d is ConstantInitializer
            conv = paddle.static.nn.conv2d(x, 5, 3)

        block = startup_prog.global_block()
        self.assertEqual(len(block.ops), 2)

        # init weight is the first op, and bias is the second
        bias_init_op = block.ops[1]
        self.assertEqual(bias_init_op.type, 'gaussian_random')
        self.assertAlmostEqual(bias_init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(bias_init_op.attr('std'), 2.0, delta=DELTA)
        self.assertEqual(bias_init_op.attr('seed'), 0)

        param_init_op = block.ops[0]
        self.assertEqual(param_init_op.type, 'uniform_random')
        self.assertAlmostEqual(param_init_op.attr('min'), -0.5, delta=DELTA)
        self.assertAlmostEqual(param_init_op.attr('max'), 0.5, delta=DELTA)
        self.assertEqual(param_init_op.attr('seed'), 0)
        base.set_global_initializer(None)


class TestKaimingUniform(unittest.TestCase):
    def func_kaiminguniform_initializer_fan_in_zero(self):
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[1, 0, 0], dtype='float32')

        kaiming = paddle.nn.initializer.KaimingUniform(0)
        param_attr = paddle.ParamAttr(initializer=kaiming)

        paddle.static.nn.prelu(x, 'all', param_attr=param_attr)

    def test_type_error(self):
        self.assertRaises(
            ZeroDivisionError, self.func_kaiminguniform_initializer_fan_in_zero
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

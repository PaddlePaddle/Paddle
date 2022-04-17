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

from __future__ import print_function

import numpy as np
import math
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.initializer as initializer
from paddle.fluid.core import VarDesc

DELTA = 0.00001


def check_cast_op(op):
    return op.type == 'cast' and \
           op.attr('in_dtype') == VarDesc.VarType.FP32 and \
           op.attr('out_dtype') in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]


def output_hist(out):
    hist, _ = np.histogram(out, range=(-1, 1))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestConstantInitializer(unittest.TestCase):
    def test_calculate_gain(self):
        self.assertEqual(paddle.nn.initializer.calculate_gain('sigmoid'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('linear'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('conv2d'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('tanh'), 5.0 / 3)
        self.assertEqual(
            paddle.nn.initializer.calculate_gain('relu'), math.sqrt(2.0))
        self.assertEqual(
            paddle.nn.initializer.calculate_gain('leaky_relu', 1), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('selu'), 3.0 / 4)

    def test_constant_initializer_default_value(self, dtype="float32"):
        """Test the constant initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.ConstantInitializer())
        num_ops = 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 0.0, delta=DELTA)
        return block

    def test_constant_initializer(self, dtype="float32"):
        """Test constant initializer with supplied value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.ConstantInitializer(2.3))
        num_ops = 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 2.3, delta=DELTA)
        return block

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16
        """
        self.test_constant_initializer_default_value("float16")
        self.test_constant_initializer("float16")

    def test_constant_initializer_bf16(self):
        """Test constant initializer with bfloat16
           No cast operator has been added here
        """
        self.test_constant_initializer_default_value("uint16")
        self.test_constant_initializer("uint16")


class TestUniformInitializer(unittest.TestCase):
    def test_uniform_initializer_default_value(self, dtype="float32"):
        """Test the uniform initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer())
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        self.assertAlmostEqual(init_op.attr('min'), -1.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), 1.0, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)
        return block

    def test_uniform_initializer_random_seed(self):
        """Test the uniform initializer with manually setting seed
        """
        program = framework.Program()
        program.random_seed = 123
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param1",
                initializer=initializer.UniformInitializer())
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param2",
                initializer=initializer.UniformInitializer(seed=456))
        init_op = block.ops[1]
        self.assertEqual(init_op.attr("seed"), 456)
        init_op1 = block.ops[0]
        self.assertEqual(init_op1.attr("seed"), 123)

    def test_uniform_initializer(self, dtype="float32"):
        """Test uniform initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer(-4.2, 3.1, 123))
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        self.assertAlmostEqual(init_op.attr('min'), -4.2, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), 3.1, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 123)
        return block

    def test_uniform_initializer_two_op(self, dtype="float32"):
        """Test uniform initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for i in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer(-4.2, float(i), 123))
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op0 = block.ops[0]
        self.assertEqual(init_op0.type, 'uniform_random')
        self.assertAlmostEqual(init_op0.attr('min'), -4.2, delta=DELTA)
        self.assertAlmostEqual(init_op0.attr('max'), 0.0, delta=DELTA)
        self.assertEqual(init_op0.attr('seed'), 123)
        return block

    def test_uniform_initializer_fp16(self):
        """Test uniform initializer with float16
        """
        block = self.test_uniform_initializer_default_value("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer(dtype="float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer_two_op("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_uniform_initializer_bf16(self):
        """Test uniform initializer with bfloat16
           No cast operator has been added here
        """
        block = self.test_uniform_initializer_default_value("uint16")
        block = self.test_uniform_initializer(dtype="uint16")
        block = self.test_uniform_initializer_two_op("uint16")


class TestNormalInitializer(unittest.TestCase):
    def test_normal_initializer_default_value(self):
        """Test the normal initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.NormalInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), 1.0, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_initializer(self, dtype="float32"):
        """Test normal initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.NormalInitializer(2.3, 1.9, 123))
        num_ops = 2 if (dtype == "float16" or dtype == "uint16") else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 123)
        return block

    def test_normal_initializer_fp16(self):
        """Test normal initializer with float16
        """
        self.test_normal_initializer("float16")

    def test_normal_initializer_bf16(self):
        """Test normal initializer with bfloat16
        """
        self.test_normal_initializer("uint16")


class TestXavierInitializer(unittest.TestCase):
    def test_uniform_xavier_initializer(self):
        """Test Xavier initializer with uniform distribution on
           for matrix multiply.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_uniform_xavier_initializer_conv(self):
        """Test Xavier initializer with uniform distribution on
           for convolutions.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10, 15, 20],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        receptive_field_size = float(15 * 20)
        limit = np.sqrt(6.0 / (
            (param.shape[0] + param.shape[1]) * receptive_field_size))
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_xavier_initializer(self):
        """Test Xavier initializer with normal distribution on
           for matrix multiply.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer(uniform=False))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        std = np.sqrt(2.0 / (param.shape[0] + param.shape[1]))
        self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_xavier_initializer_conv(self):
        """Test Xavier initializer with normal distribution on
           for convolutions.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10, 15, 20],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer(uniform=False))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        receptive_field_size = float(15 * 20)
        std = np.sqrt(2.0 / (
            (param.shape[0] + param.shape[1]) * receptive_field_size))
        self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_xavier_initializer_supplied_arguments(self,
                                                   dtype="float32",
                                                   uniform=True):
        """Test the Xavier initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer(
                    uniform=uniform, fan_in=12, fan_out=23, seed=134))
        num_ops = 2 if (dtype == "float16" or (dtype == "uint16" and
                                               not uniform)) else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        if uniform:
            self.assertEqual(init_op.type, 'uniform_random')
            limit = np.sqrt(6.0 / (12 + 23))
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        else:
            self.assertEqual(init_op.type, 'gaussian_random')
        self.assertEqual(init_op.attr('seed'), 134)
        return block

    def test_xavier_initializer_fp16(self):
        """Test the Xavier initializer with float16
        """
        block = self.test_xavier_initializer_supplied_arguments("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_xavier_initializer_bf16(self):
        """Test the Xavier initializer with bfloat16
        """
        block_uniform = self.test_xavier_initializer_supplied_arguments(
            "uint16")
        self.assertEqual(len(block_uniform.ops), 1)
        block_gaussian = self.test_xavier_initializer_supplied_arguments(
            "uint16", False)
        self.assertTrue(check_cast_op(block_gaussian.ops[1]))


class TestMSRAInitializer(unittest.TestCase):
    def test_uniform_msra_initializer(self):
        """Test MSRA initializer with uniform distribution on
           for matrix multiply.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / param.shape[0])
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_uniform_msra_initializer_conv(self):
        """Test MSRA initializer with uniform distribution on
           for convolutions.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10, 15, 20],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        receptive_field_size = float(15 * 20)
        limit = np.sqrt(6.0 / (param.shape[1] * receptive_field_size))
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_msra_initializer(self):
        """Test MSRA initializer with normal distribution on
           for matrix multiply.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer(uniform=False))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        std = np.sqrt(2.0 / param.shape[0])
        self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_msra_initializer_conv(self):
        """Test MSRA initializer with normal distribution on
           for convolutions.
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            param = block.create_parameter(
                dtype="float32",
                shape=[5, 10, 15, 20],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer(uniform=False))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        receptive_field_size = float(15 * 20)
        std = np.sqrt(2.0 / (param.shape[1] * receptive_field_size))
        self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

    def test_msra_initializer_supplied_arguments(self, dtype="float32"):
        """Test the MSRA initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer(
                    fan_in=12, seed=134))
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / 12)
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 134)
        return block

    def test_msra_initializer_fp16(self):
        """Test the MSRA initializer with float16
        """
        block = self.test_msra_initializer_supplied_arguments("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_msra_initializer_bf16(self):
        """Test the MSRA initializer with bfloat16
        """
        block = self.test_msra_initializer_supplied_arguments("uint16")


class TestBilinearInitializer(unittest.TestCase):
    def test_bilinear_initializer(self, dtype="float32"):
        """Test the bilinear initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[8, 1, 3, 3],
                lod_level=0,
                name="param",
                initializer=initializer.BilinearInitializer())
        num_ops = 2 if dtype in ["float16", "uint16", "float64"] else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'assign_value')
        return block

    def test_bilinear_initializer_fp64(self):
        self.test_bilinear_initializer(dtype='float64')

    def test_bilinear_initializer_fp16(self):
        """Test the bilinear initializer with supplied arguments
        """
        block = self.test_bilinear_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_bilinear_initializer_bf16(self):
        """Test the bilinear initializer with supplied arguments
        """
        block = self.test_bilinear_initializer("uint16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_type_error(self):
        self.assertRaises(TypeError, self.test_bilinear_initializer, 'int32')


class TestNumpyArrayInitializer(unittest.TestCase):
    def test_numpy_array_initializer(self, dtype="float32"):
        """Test the numpy array initializer with supplied arguments
        """
        import numpy
        program = framework.Program()
        block = program.global_block()
        np_array = numpy.random.random((10000)).astype(dtype)
        for _ in range(2):
            block.create_parameter(
                dtype=np_array.dtype,
                shape=np_array.shape,
                lod_level=0,
                name="param",
                initializer=initializer.NumpyArrayInitializer(np_array))
        num_ops = 2 if dtype in ["float16", "uint16"] else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'assign_value')
        assert (init_op.attr('fp32_values') == np_array).all()
        return block

    def test_numpy_array_initializer_fp16(self):
        """Test the numpy array initializer with float16
        """
        block = self.test_numpy_array_initializer("float16")
        self.assertTrue(block.ops[1])

    def test_numpy_array_initializer_bf16(self):
        """Test the numpy array initializer with bfloat16
        """
        block = self.test_numpy_array_initializer("uint16")
        self.assertTrue(block.ops[1])


class TestSetGlobalInitializer(unittest.TestCase):
    def test_set_global_weight_initilizer(self):
        """Test Set Global Param initilizer with UniformInitializer
        """
        main_prog = framework.Program()
        startup_prog = framework.Program()
        fluid.set_global_initializer(initializer.Uniform(low=-0.5, high=0.5))
        with fluid.program_guard(main_prog, startup_prog):
            x = fluid.data(name="x", shape=[1, 3, 32, 32])
            # default initilizer of param in layers.conv2d is NormalInitializer
            conv = fluid.layers.conv2d(x, 5, 3)

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
        fluid.set_global_initializer(None)

    def test_set_global_bias_initilizer(self):
        """Test Set Global Bias initilizer with NormalInitializer
        """
        main_prog = framework.Program()
        startup_prog = framework.Program()
        fluid.set_global_initializer(
            initializer.Uniform(
                low=-0.5, high=0.5),
            bias_init=initializer.Normal(
                loc=0.0, scale=2.0))
        with fluid.program_guard(main_prog, startup_prog):
            x = fluid.data(name="x", shape=[1, 3, 32, 32])
            # default initilizer of bias in layers.conv2d is ConstantInitializer
            conv = fluid.layers.conv2d(x, 5, 3)

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
        fluid.set_global_initializer(None)


class TestUniformInitializerDygraph(unittest.TestCase):
    def func_uniform_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False
        self.assertTrue(np.allclose(np.zeros((1024, 1024, 16)), tensor.numpy()))

        uniform_ = paddle.nn.initializer.Uniform()
        uniform_(tensor)

        self.assertEqual(tensor.stop_gradient,
                         False)  # stop_gradient is not changed

        hist, prob = output_hist(tensor.numpy())

        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=1e-3), "hist: " + str(hist))

        paddle.enable_static()

    def test_uniform_initializer(self, dtype="float32"):
        with framework._test_eager_guard():
            self.func_uniform_initializer()
        self.func_uniform_initializer()


class TestXavierInitializerDygraph(unittest.TestCase):
    def func_xvarier_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False

        xavier_ = paddle.fluid.initializer.XavierInitializer(
            uniform=False, fan_in=3, fan_out=5)
        xavier_(tensor)

        hist, _ = output_hist(tensor.numpy())

        hist2, _ = output_hist(
            np.random.normal(0, np.sqrt(2.0 / (3 + 5)), [1024, 1024, 16]))

        self.assertTrue(
            np.allclose(
                hist, hist2, rtol=0, atol=0.01),
            "hist: " + str(hist) + " hist2: " + str(hist2))
        paddle.enable_static()

    def test_xavier_initializer(self, dtype="float32"):
        with framework._test_eager_guard():
            self.func_xvarier_initializer()
        self.func_xvarier_initializer()


class TestMSRAInitializerDygraph(unittest.TestCase):
    def func_msra_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False

        msra_ = paddle.fluid.initializer.MSRAInitializer(
            uniform=False, fan_in=4)
        msra_(tensor)

        hist, _ = output_hist(tensor.numpy())

        hist2, _ = output_hist(
            np.random.normal(0, np.sqrt(2.0 / (4)), [1024, 1024, 16]))

        self.assertTrue(
            np.allclose(
                hist, hist2, rtol=0, atol=0.01),
            "hist: " + str(hist) + " hist2: " + str(hist2))
        paddle.enable_static()

    def test_msra_initializer(self, dtype="float32"):
        with framework._test_eager_guard():
            self.func_msra_initializer()
        self.func_msra_initializer()


class TesetconsistencyOfDynamicAndStaticGraph(unittest.TestCase):
    def func_order(self):
        paddle.set_device('cpu')
        SEED = 123
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=2.0))
        bias_attr = paddle.framework.ParamAttr(
            name="linear_bias",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=2.0))

        def run_dynamic_graph():
            paddle.disable_static()
            paddle.seed(SEED)
            linear = paddle.nn.Linear(
                1, 1, weight_attr=weight_attr, bias_attr=bias_attr)
            return linear.weight.numpy(), linear.bias.numpy()
            paddle.enable_static()

        def run_static_graph():
            paddle.enable_static()
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle.seed(SEED)
            linear = paddle.nn.Linear(
                1, 1, weight_attr=weight_attr, bias_attr=bias_attr)
            res = exe.run(paddle.static.default_startup_program(),
                          fetch_list=['linear_weight', 'linear_bias'])
            return res[0], res[1]

        dynamic_res = run_dynamic_graph()
        static_res = run_static_graph()

        self.assertTrue(np.array_equal(dynamic_res[0], static_res[0]))
        self.assertTrue(np.array_equal(dynamic_res[1], static_res[1]))

    def test_order(self):
        with framework._test_eager_guard():
            self.func_order()
        self.func_order()


# 2-D Parameter with shape: [10, 15]
class TestOrthogonalInitializer1(unittest.TestCase):
    """
    case 1
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=3.0))
        self.dtype = "float64"
        self.in_features = 10
        self.out_features = 15
        self.num_ops = 9

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(np.matmul(a, a.T), 9 * np.eye(10)))

    def func_orthogonal(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        paddle.seed(2021)
        linear = paddle.nn.Linear(
            self.in_features, self.out_features, weight_attr=self.weight_attr)
        res_dygraph = linear.weight.numpy()

        paddle.enable_static()
        paddle.seed(2021)
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            linear = paddle.nn.Linear(
                self.in_features,
                self.out_features,
                weight_attr=self.weight_attr)

            block = start_prog.global_block()
            self.assertEqual(len(block.ops), self.num_ops)
            self.assertEqual(block.ops[0].type, 'gaussian_random')
            self.assertEqual(block.ops[1].type, 'qr')
            self.assertEqual(block.ops[2].type, 'diag_v2')
            self.assertEqual(block.ops[3].type, 'sign')
            self.assertEqual(block.ops[4].type, 'elementwise_mul')
            self.assertEqual(block.ops[-3].type, 'reshape2')
            self.assertEqual(block.ops[-2].type, 'scale')

            exe = paddle.static.Executor()
            res_static = exe.run(start_prog, fetch_list=[linear.weight])[0]

        self.check_result(res_dygraph, res_static)

    def test_orthogonal(self):
        with framework._test_eager_guard():
            self.func_orthogonal()
        self.func_orthogonal()


# 2-D Parameter with shape: [15, 10]
class TestOrthogonalInitializer2(TestOrthogonalInitializer1):
    """
    case 2
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=2.0))
        self.dtype = "float64"
        self.in_features = 15
        self.out_features = 10
        self.num_ops = 8

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(np.matmul(a.T, a), 4 * np.eye(10)))


# 2-D Parameter with shape: [10, 10]
class TestOrthogonalInitializer3(TestOrthogonalInitializer1):
    """
    case 3
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal())
        self.dtype = "float32"
        self.in_features = 10
        self.out_features = 10
        self.num_ops = 8

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(np.matmul(a.T, a), np.eye(10), atol=1.e-6))
        self.assertTrue(np.allclose(np.matmul(a, a.T), np.eye(10), atol=1.e-6))

    def test_error(self):
        self.config()
        with self.assertRaises(AssertionError):
            paddle.nn.Linear(10, 10, bias_attr=self.weight_attr)


# 4-D Parameter with shape: [6, 4, 3, 3]
class TestOrthogonalInitializer4(unittest.TestCase):
    """
    case 4
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=3.0))
        self.dtype = "float64"
        self.in_features = 4
        self.out_features = 6
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        a = a.reshape(6, -1)
        self.assertTrue(np.allclose(np.matmul(a, a.T), 9 * np.eye(6)))

    def func_orthogonal(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        paddle.seed(2021)
        conv2d = paddle.nn.Conv2D(
            self.in_features,
            self.out_features,
            self.kernel_size,
            weight_attr=self.weight_attr)
        res_dygraph = conv2d.weight.numpy()

        paddle.enable_static()
        paddle.seed(2021)
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            conv2d = paddle.nn.Conv2D(
                self.in_features,
                self.out_features,
                self.kernel_size,
                weight_attr=self.weight_attr)
            exe = paddle.static.Executor()
            res_static = exe.run(paddle.static.default_startup_program(),
                                 fetch_list=[conv2d.weight])[0]
        self.check_result(res_dygraph, res_static)

    def test_orthogonal(self):
        with framework._test_eager_guard():
            self.func_orthogonal()
        self.func_orthogonal()


# 4-D Parameter with shape: [50, 4, 3, 3]
class TestOrthogonalInitializer5(TestOrthogonalInitializer4):
    """
    case 5
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=2.0))
        self.dtype = "float64"
        self.in_features = 4
        self.out_features = 50
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        a = a.reshape(50, -1)
        self.assertTrue(np.allclose(np.matmul(a.T, a), 4 * np.eye(36)))


# 4-D Parameter with shape: [36, 4, 3, 3]
class TestOrthogonalInitializer6(TestOrthogonalInitializer4):
    """
    case 6
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal())
        self.dtype = "float32"
        self.in_features = 4
        self.out_features = 36
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        self.assertTrue(np.array_equal(a, b))
        a = a.reshape(36, -1)
        self.assertTrue(np.allclose(np.matmul(a.T, a), np.eye(36), atol=1.e-6))
        self.assertTrue(np.allclose(np.matmul(a, a.T), np.eye(36), atol=1.e-6))


# initialize Conv1D weight
class TestDiracInitializer1(unittest.TestCase):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac())
        self.dtype = "float64"
        self.in_channels = 3
        self.out_channels = 2
        self.kernel_size = 3
        self.input_shape = [8, self.in_channels, 10]
        self.conv_layer = paddle.nn.Conv1D
        self.num_ops = 8  #fill_constant*2, reshape*2, assign_value*2, scatter, cast

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        self.assertTrue(np.array_equal(w_dygraph, w_static))
        self.assertTrue(np.array_equal(conv_out, conv_in[:, 0:2, 1:9]))

    def func_dirac(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        conv = self.conv_layer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            weight_attr=self.weight_attr)
        weight_dygraph = conv.weight.numpy()

        paddle.enable_static()
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            inp = paddle.rand(self.input_shape)
            conv = self.conv_layer(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                weight_attr=self.weight_attr)

            output = conv(inp)
            block = start_prog.global_block()
            self.assertEqual(len(block.ops), self.num_ops)
            self.assertEqual(block.ops[0].type, 'fill_constant')
            self.assertEqual(block.ops[1].type, 'reshape')
            self.assertEqual(block.ops[2].type, 'assign_value')
            self.assertEqual(block.ops[3].type, 'assign_value')
            self.assertEqual(block.ops[4].type, 'scatter')
            self.assertEqual(block.ops[5].type, 'reshape')

            exe = paddle.static.Executor()
            exe.run(start_prog)
            fetch = exe.run(main_prog, fetch_list=[inp, output, conv.weight])
            conv_input = fetch[0]
            conv_output = fetch[1]
            weight_static = fetch[2]

        self.check_result(weight_dygraph, weight_static, conv_input,
                          conv_output)

    def test_dirac(self):
        with framework._test_eager_guard():
            self.func_dirac()
        self.func_dirac()


# initialize Conv2D weight
class TestDiracInitializer2(TestDiracInitializer1):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac(groups=1))
        self.dtype = "float64"
        self.in_channels = 4
        self.out_channels = 8
        self.kernel_size = (3, 3)
        self.input_shape = [8, self.in_channels, 10, 10]
        self.conv_layer = paddle.nn.Conv2D
        self.num_ops = 8

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        self.assertTrue(np.array_equal(w_dygraph, w_static))
        self.assertTrue(
            np.array_equal(conv_out[:, 0:4, :, :], conv_in[:, :, 1:9, 1:9]))
        self.assertTrue(
            np.array_equal(conv_out[:, 4:8, :, :], np.zeros([8, 4, 8, 8])))


# initialize Conv3D weight
class TestDiracInitializer3(TestDiracInitializer1):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac(groups=2))
        self.dtype = "float32"
        self.in_channels = 5
        self.out_channels = 10
        self.kernel_size = (3, 3, 3)
        self.input_shape = [8, self.in_channels, 10, 10, 10]
        self.conv_layer = paddle.nn.Conv3D
        self.num_ops = 7

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        self.assertTrue(np.array_equal(w_dygraph, w_static))
        self.assertTrue(
            np.array_equal(conv_out[:, 0:5, :, :, :], conv_in[:, :, 1:9, 1:9, 1:
                                                              9]))
        self.assertTrue(
            np.array_equal(conv_out[:, 5:10, :, :, :], conv_in[:, :, 1:9, 1:9,
                                                               1:9]))

    def test_error(self):
        self.config()
        with self.assertRaises(AssertionError):
            paddle.nn.Linear(10, 10, weight_attr=self.weight_attr)

        with self.assertRaises(AssertionError):
            paddle.nn.Conv2D(5, 9, (3, 3), weight_attr=self.weight_attr)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

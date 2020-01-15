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
import unittest

import paddle.fluid.framework as framework
import paddle.fluid.initializer as initializer
from paddle.fluid.core import VarDesc

DELTA = 0.00001


def check_cast_op(op):
    return op.type == 'cast' and \
           op.attr('in_dtype') == VarDesc.VarType.FP32 and \
           op.attr('out_dtype') == VarDesc.VarType.FP16


class TestConstantInitializer(unittest.TestCase):
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
        num_ops = 2 if dtype == "float16" else 1
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
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 2.3, delta=DELTA)
        return block

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16
        """
        block = self.test_constant_initializer_default_value("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_constant_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))


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
        self.assertEqual(init_op.attr("seed"), 123)
        init_op1 = block.ops[0]
        self.assertEqual(init_op1.attr("seed"), 456)

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
        num_ops = 2 if dtype == "float16" else 1
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
        block = self.test_normal_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))


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

    def test_xavier_initializer_supplied_arguments(self, dtype="float32"):
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
                    fan_in=12, fan_out=23, seed=134))
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / (12 + 23))
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 134)
        return block

    def test_xavier_initializer_fp16(self):
        """Test the Xavier initializer with float16
        """
        block = self.test_xavier_initializer_supplied_arguments("float16")
        self.assertTrue(check_cast_op(block.ops[1]))


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
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'assign_value')
        return block

    def test_bilinear_initializer_fp16(self):
        """Test the bilinear initializer with supplied arguments
        """
        block = self.test_bilinear_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))


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
        num_ops = 2 if dtype == "float16" else 1
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


if __name__ == '__main__':
    unittest.main()

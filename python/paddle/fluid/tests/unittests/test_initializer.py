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

DELTA = 0.00001


class TestConstantInitializer(unittest.TestCase):
    def test_constant_initializer_default_value(self):
        """Test the constant initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.ConstantInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 0.0, delta=DELTA)

    def test_constant_initializer(self):
        """Test constant initializer with supplied value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.ConstantInitializer(2.3))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 2.3, delta=DELTA)


class TestUniformInitializer(unittest.TestCase):
    def test_uniform_initializer_default_value(self):
        """Test the uniform initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        self.assertAlmostEqual(init_op.attr('min'), -1.0, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), 1.0, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 0)

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

    def test_uniform_initializer(self):
        """Test uniform initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer(-4.2, 3.1, 123))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        self.assertAlmostEqual(init_op.attr('min'), -4.2, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), 3.1, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 123)

    def test_uniform_initializer_two_op(self):
        """Test uniform initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for i in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.UniformInitializer(-4.2, float(i), 123))
        self.assertEqual(len(block.ops), 1)
        init_op0 = block.ops[0]
        self.assertEqual(init_op0.type, 'uniform_random')
        self.assertAlmostEqual(init_op0.attr('min'), -4.2, delta=DELTA)
        self.assertAlmostEqual(init_op0.attr('max'), 0.0, delta=DELTA)
        self.assertEqual(init_op0.attr('seed'), 123)


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

    def test_normal_initializer(self):
        """Test normal initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.NormalInitializer(2.3, 1.9, 123))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'gaussian_random')
        self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 123)


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

    def test_xavier_initializer_supplied_arguments(self):
        """Test the Xavier initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.XavierInitializer(
                    fan_in=12, fan_out=23, seed=134))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / (12 + 23))
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 134)


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

    def test_msra_initializer_supplied_arguments(self):
        """Test the MSRA initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.MSRAInitializer(
                    fan_in=12, seed=134))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'uniform_random')
        limit = np.sqrt(6.0 / 12)
        self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
        self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
        self.assertEqual(init_op.attr('seed'), 134)


class TestMSRAInitializer(unittest.TestCase):
    def test_bilinear_initializer(self):
        """Test the bilinear initializer with supplied arguments
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype="float32",
                shape=[8, 1, 3, 3],
                lod_level=0,
                name="param",
                initializer=initializer.BilinearInitializer())
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'assign_value')


class TestNumpyArrayInitializer(unittest.TestCase):
    def test_numpy_array_initializer(self):
        """Test the numpy array initializer with supplied arguments
        """
        import numpy
        program = framework.Program()
        block = program.global_block()
        np_array = numpy.random.random((10000)).astype("float32")
        for _ in range(2):
            block.create_parameter(
                dtype=np_array.dtype,
                shape=np_array.shape,
                lod_level=0,
                name="param",
                initializer=initializer.NumpyArrayInitializer(np_array))
        self.assertEqual(len(block.ops), 1)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'assign_value')
        assert (init_op.attr('fp32_values') == np_array).all()


if __name__ == '__main__':
    unittest.main()

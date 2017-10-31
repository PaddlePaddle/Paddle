import unittest

import paddle.v2.framework.framework as framework
import paddle.v2.framework.initializer as initializer

DELTA = 0.00001


class TestConstantInitializer(unittest.TestCase):
    def test_constant_initializer_default_value(self):
        """Test the constant initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
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

    def test_uniform_initializer(self):
        """Test uniform initializer with supplied attributes
        """
        program = framework.Program()
        block = program.global_block()
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


class TestNormalInitializer(unittest.TestCase):
    def test_normal_initializer_default_value(self):
        """Test the normal initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
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


if __name__ == '__main__':
    unittest.main()

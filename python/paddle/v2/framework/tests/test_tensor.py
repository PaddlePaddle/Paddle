import paddle.v2.framework.core as core
import unittest
import numpy


class TestScope(unittest.TestCase):
    def test_int_tensor(self):
        scope = core.Scope()
        var = scope.new_var("test_tensor")
        tensor = var.get_tensor()

        tensor.set_dims([1000, 784])
        tensor.alloc_int()

        tensor_array = numpy.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1
        tensor_array[19, 11] = 2
        tensor.set(tensor_array)

        tensor_array_2 = numpy.array(tensor)
        self.assertEqual(1.0, tensor_array_2[3, 9])
        self.assertEqual(2.0, tensor_array_2[19, 11])

    def test_float_tensor(self):
        scope = core.Scope()
        var = scope.new_var("test_tensor")
        tensor = var.get_tensor()

        tensor.set_dims([1000, 784])
        tensor.alloc_float()

        tensor_array = numpy.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1.0
        tensor_array[19, 11] = 2.0
        tensor.set(tensor_array)

        tensor_array_2 = numpy.array(tensor)
        self.assertAlmostEqual(1.0, tensor_array_2[3, 9])
        self.assertAlmostEqual(2.0, tensor_array_2[19, 11])


if __name__ == '__main__':
    unittest.main()

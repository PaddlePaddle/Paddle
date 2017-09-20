import paddle.v2.framework.core as core
import unittest
import numpy


class TestTensor(unittest.TestCase):
    def test_int_tensor(self):
        scope = core.Scope()
        var = scope.new_var("test_tensor")
        place = core.CPUPlace()

        tensor = var.get_tensor()

        tensor.set_dims([1000, 784])
        tensor.alloc_int(place)
        tensor_array = numpy.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1
        tensor_array[19, 11] = 2
        tensor.set(tensor_array, place)

        tensor_array_2 = numpy.array(tensor)
        self.assertEqual(1, tensor_array_2[3, 9])
        self.assertEqual(2, tensor_array_2[19, 11])

    def test_float_tensor(self):
        scope = core.Scope()
        var = scope.new_var("test_tensor")
        place = core.CPUPlace()

        tensor = var.get_tensor()

        tensor.set_dims([1000, 784])
        tensor.alloc_float(place)

        tensor_array = numpy.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1.0
        tensor_array[19, 11] = 2.0
        tensor.set(tensor_array, place)

        tensor_array_2 = numpy.array(tensor)
        self.assertAlmostEqual(1.0, tensor_array_2[3, 9])
        self.assertAlmostEqual(2.0, tensor_array_2[19, 11])

    def test_int_lod_tensor(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var_lod = scope.new_var("test_lod_tensor")
        lod_tensor = var_lod.get_tensor()

        lod_tensor.set_dims([4, 4, 6])
        lod_tensor.alloc_int(place)
        array = numpy.array(lod_tensor)
        array[0, 0, 0] = 3
        array[3, 3, 5] = 10
        lod_tensor.set(array, place)
        lod_tensor.set_lod([[0, 2, 4]])

        lod_v = numpy.array(lod_tensor)
        self.assertTrue(numpy.alltrue(array == lod_v))

        lod = lod_tensor.lod()
        self.assertEqual(0, lod[0][0])
        self.assertEqual(2, lod[0][1])
        self.assertEqual(4, lod[0][2])

    def test_float_lod_tensor(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var_lod = scope.new_var("test_lod_tensor")

        lod_tensor = var_lod.get_tensor()
        lod_tensor.set_dims([5, 2, 3, 4])
        lod_tensor.alloc_float(place)

        tensor_array = numpy.array(lod_tensor)
        self.assertEqual((5, 2, 3, 4), tensor_array.shape)
        tensor_array[0, 0, 0, 0] = 1.0
        tensor_array[0, 0, 0, 1] = 2.0
        lod_tensor.set(tensor_array, place)

        lod_v = numpy.array(lod_tensor)
        self.assertAlmostEqual(1.0, lod_v[0, 0, 0, 0])
        self.assertAlmostEqual(2.0, lod_v[0, 0, 0, 1])
        self.assertEqual(len(lod_tensor.lod()), 0)

        lod_py = [[0, 2, 5], [0, 2, 4, 5]]
        lod_tensor.set_lod(lod_py)
        lod = lod_tensor.lod()
        self.assertListEqual(lod_py, lod)

    def test_lod_tensor_init(self):
        scope = core.Scope()
        place = core.CPUPlace()
        lod_py = [[0, 2, 5], [0, 2, 4, 5]]
        lod_tensor = core.LoDTensor(lod_py)

        lod_tensor.set_dims([5, 2, 3, 4])
        lod_tensor.alloc_float(place)
        tensor_array = numpy.array(lod_tensor)
        tensor_array[0, 0, 0, 0] = 1.0
        tensor_array[0, 0, 0, 1] = 2.0
        lod_tensor.set(tensor_array, place)

        lod_v = numpy.array(lod_tensor)
        self.assertAlmostEqual(1.0, lod_v[0, 0, 0, 0])
        self.assertAlmostEqual(2.0, lod_v[0, 0, 0, 1])
        self.assertListEqual(lod_py, lod_tensor.lod())


if __name__ == '__main__':
    unittest.main()

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

import paddle.fluid as fluid
import paddle.fluid.core as core
import unittest
import numpy as np
import numbers


class TestTensorPtr(unittest.TestCase):

    def test_tensor_ptr(self):
        t = core.Tensor()
        np_arr = np.zeros([2, 3])
        t.set(np_arr, core.CPUPlace())
        self.assertGreater(t._ptr(), 0)


class TestTensor(unittest.TestCase):

    def setUp(self):
        self.support_dtypes = [
            'bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16',
            'float32', 'float64'
        ]

    def test_int_tensor(self):
        scope = core.Scope()
        var = scope.var("test_tensor")
        place = core.CPUPlace()

        tensor = var.get_tensor()

        tensor._set_dims([1000, 784])
        tensor._alloc_int(place)
        tensor_array = np.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1
        tensor_array[19, 11] = 2
        tensor.set(tensor_array, place)

        tensor_array_2 = np.array(tensor)
        self.assertEqual(1, tensor_array_2[3, 9])
        self.assertEqual(2, tensor_array_2[19, 11])

    def test_float_tensor(self):
        scope = core.Scope()
        var = scope.var("test_tensor")
        place = core.CPUPlace()

        tensor = var.get_tensor()

        tensor._set_dims([1000, 784])
        tensor._alloc_float(place)

        tensor_array = np.array(tensor)
        self.assertEqual((1000, 784), tensor_array.shape)
        tensor_array[3, 9] = 1.0
        tensor_array[19, 11] = 2.0
        tensor.set(tensor_array, place)

        tensor_array_2 = np.array(tensor)
        self.assertAlmostEqual(1.0, tensor_array_2[3, 9])
        self.assertAlmostEqual(2.0, tensor_array_2[19, 11])

    def test_int8_tensor(self):
        scope = core.Scope()
        var = scope.var("int8_tensor")
        cpu_tensor = var.get_tensor()
        tensor_array = np.random.randint(-127,
                                         high=128,
                                         size=[100, 200],
                                         dtype=np.int8)
        place = core.CPUPlace()
        cpu_tensor.set(tensor_array, place)
        cpu_tensor_array_2 = np.array(cpu_tensor)
        self.assertAlmostEqual(cpu_tensor_array_2.all(), tensor_array.all())

        if core.is_compiled_with_cuda():
            cuda_tensor = var.get_tensor()
            tensor_array = np.random.randint(-127,
                                             high=128,
                                             size=[100, 200],
                                             dtype=np.int8)
            place = core.CUDAPlace(0)
            cuda_tensor.set(tensor_array, place)
            cuda_tensor_array_2 = np.array(cuda_tensor)
            self.assertAlmostEqual(cuda_tensor_array_2.all(),
                                   tensor_array.all())

    def test_int_lod_tensor(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var_lod = scope.var("test_lod_tensor")
        lod_tensor = var_lod.get_tensor()

        lod_tensor._set_dims([4, 4, 6])
        lod_tensor._alloc_int(place)
        array = np.array(lod_tensor)
        array[0, 0, 0] = 3
        array[3, 3, 5] = 10
        lod_tensor.set(array, place)
        lod_tensor.set_recursive_sequence_lengths([[2, 2]])

        lod_v = np.array(lod_tensor)
        self.assertTrue(np.alltrue(array == lod_v))

        lod = lod_tensor.recursive_sequence_lengths()
        self.assertEqual(2, lod[0][0])
        self.assertEqual(2, lod[0][1])

    def test_float_lod_tensor(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var_lod = scope.var("test_lod_tensor")

        lod_tensor = var_lod.get_tensor()
        lod_tensor._set_dims([5, 2, 3, 4])
        lod_tensor._alloc_float(place)

        tensor_array = np.array(lod_tensor)
        self.assertEqual((5, 2, 3, 4), tensor_array.shape)
        tensor_array[0, 0, 0, 0] = 1.0
        tensor_array[0, 0, 0, 1] = 2.0
        lod_tensor.set(tensor_array, place)

        lod_v = np.array(lod_tensor)
        self.assertAlmostEqual(1.0, lod_v[0, 0, 0, 0])
        self.assertAlmostEqual(2.0, lod_v[0, 0, 0, 1])
        self.assertEqual(len(lod_tensor.recursive_sequence_lengths()), 0)

        lod_py = [[2, 1], [1, 2, 2]]
        lod_tensor.set_recursive_sequence_lengths(lod_py)
        lod = lod_tensor.recursive_sequence_lengths()
        self.assertListEqual(lod_py, lod)

    def test_lod_tensor_init(self):
        place = core.CPUPlace()
        lod_py = [[2, 1], [1, 2, 2]]
        lod_tensor = core.LoDTensor()

        lod_tensor._set_dims([5, 2, 3, 4])
        lod_tensor.set_recursive_sequence_lengths(lod_py)
        lod_tensor._alloc_float(place)
        tensor_array = np.array(lod_tensor)
        tensor_array[0, 0, 0, 0] = 1.0
        tensor_array[0, 0, 0, 1] = 2.0
        lod_tensor.set(tensor_array, place)

        lod_v = np.array(lod_tensor)
        self.assertAlmostEqual(1.0, lod_v[0, 0, 0, 0])
        self.assertAlmostEqual(2.0, lod_v[0, 0, 0, 1])
        self.assertListEqual(lod_py, lod_tensor.recursive_sequence_lengths())

    def test_lod_tensor_gpu_init(self):
        if not core.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        lod_py = [[2, 1], [1, 2, 2]]
        lod_tensor = core.LoDTensor()

        lod_tensor._set_dims([5, 2, 3, 4])
        lod_tensor.set_recursive_sequence_lengths(lod_py)
        lod_tensor._alloc_float(place)
        tensor_array = np.array(lod_tensor)
        tensor_array[0, 0, 0, 0] = 1.0
        tensor_array[0, 0, 0, 1] = 2.0
        lod_tensor.set(tensor_array, place)

        lod_v = np.array(lod_tensor)
        self.assertAlmostEqual(1.0, lod_v[0, 0, 0, 0])
        self.assertAlmostEqual(2.0, lod_v[0, 0, 0, 1])
        self.assertListEqual(lod_py, lod_tensor.recursive_sequence_lengths())

    def test_empty_tensor(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var = scope.var("test_tensor")

        tensor = var.get_tensor()
        tensor._set_dims([0, 1])
        tensor._alloc_float(place)

        tensor_array = np.array(tensor)
        self.assertEqual((0, 1), tensor_array.shape)

        if core.is_compiled_with_cuda():
            gpu_place = core.CUDAPlace(0)
            tensor._alloc_float(gpu_place)
            tensor_array = np.array(tensor)
            self.assertEqual((0, 1), tensor_array.shape)

    def run_slice_tensor(self, place, dtype):
        tensor = fluid.Tensor()
        shape = [3, 3, 3]
        tensor._set_dims(shape)

        tensor_array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                                 [[19, 20, 21], [22, 23, 24],
                                  [25, 26, 27]]]).astype(dtype)

        tensor.set(tensor_array, place)
        n1 = tensor[1]
        t1 = tensor_array[1]
        self.assertTrue((np.array(n1) == np.array(t1)).all())

        n2 = tensor[1:]
        t2 = tensor_array[1:]
        self.assertTrue((np.array(n2) == np.array(t2)).all())

        n3 = tensor[0:2:]
        t3 = tensor_array[0:2:]
        self.assertTrue((np.array(n3) == np.array(t3)).all())

        n4 = tensor[2::-2]
        t4 = tensor_array[2::-2]
        self.assertTrue((np.array(n4) == np.array(t4)).all())

        n5 = tensor[2::-2][0]
        t5 = tensor_array[2::-2][0]
        self.assertTrue((np.array(n5) == np.array(t5)).all())

        n6 = tensor[2:-1:-1]
        t6 = tensor_array[2:-1:-1]
        self.assertTrue((np.array(n6) == np.array(t6)).all())

        n7 = tensor[0:, 0:]
        t7 = tensor_array[0:, 0:]
        self.assertTrue((np.array(n7) == np.array(t7)).all())

        n8 = tensor[0::1, 0::-1, 2:]
        t8 = tensor_array[0::1, 0::-1, 2:]
        self.assertTrue((np.array(n8) == np.array(t8)).all())

    def test_slice_tensor(self):
        for dtype in self.support_dtypes:
            # run cpu first
            place = core.CPUPlace()
            self.run_slice_tensor(place, dtype)

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                self.run_slice_tensor(place, dtype)

    def test_print_tensor(self):
        scope = core.Scope()
        var = scope.var("test_tensor")
        place = core.CPUPlace()
        tensor = var.get_tensor()
        tensor._set_dims([10, 10])
        tensor._alloc_int(place)
        tensor_array = np.array(tensor)
        self.assertEqual((10, 10), tensor_array.shape)
        tensor_array[0, 0] = 1
        tensor_array[2, 2] = 2
        tensor.set(tensor_array, place)
        print(tensor)
        self.assertTrue(isinstance(str(tensor), str))

        if core.is_compiled_with_cuda():
            tensor.set(tensor_array, core.CUDAPlace(0))
            print(tensor)
            self.assertTrue(isinstance(str(tensor), str))

    def test_tensor_poiter(self):
        place = core.CPUPlace()
        scope = core.Scope()
        var = scope.var("test_tensor")
        place = core.CPUPlace()
        tensor = var.get_tensor()
        dtype = core.VarDesc.VarType.FP32
        self.assertTrue(
            isinstance(tensor._mutable_data(place, dtype), numbers.Integral))

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.assertTrue(
                isinstance(tensor._mutable_data(place, dtype),
                           numbers.Integral))
            place = core.CUDAPinnedPlace()
            self.assertTrue(
                isinstance(tensor._mutable_data(place, dtype),
                           numbers.Integral))
            places = fluid.cuda_pinned_places()
            self.assertTrue(
                isinstance(tensor._mutable_data(places[0], dtype),
                           numbers.Integral))

    def test_tensor_set_fp16(self):
        array = np.random.random((300, 500)).astype("float16")
        tensor = fluid.Tensor()
        place = core.CPUPlace()
        tensor.set(array, place)
        self.assertEqual(tensor._dtype(), core.VarDesc.VarType.FP16)
        np.testing.assert_array_equal(np.array(tensor), array)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            tensor.set(array, place)
            self.assertEqual(tensor._dtype(), core.VarDesc.VarType.FP16)
            np.testing.assert_array_equal(np.array(tensor), array)

            place = core.CUDAPinnedPlace()
            tensor.set(array, place)
            self.assertEqual(tensor._dtype(), core.VarDesc.VarType.FP16)
            np.testing.assert_array_equal(np.array(tensor), array)

    def test_tensor_set_int16(self):
        array = np.random.randint(100, size=(300, 500)).astype("int16")
        tensor = fluid.Tensor()
        place = core.CPUPlace()
        tensor.set(array, place)
        self.assertEqual(tensor._dtype(), core.VarDesc.VarType.INT16)
        np.testing.assert_array_equal(np.array(tensor), array)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            tensor.set(array, place)
            self.assertEqual(tensor._dtype(), core.VarDesc.VarType.INT16)
            np.testing.assert_array_equal(np.array(tensor), array)

            place = core.CUDAPinnedPlace()
            tensor.set(array, place)
            self.assertEqual(tensor._dtype(), core.VarDesc.VarType.INT16)
            np.testing.assert_array_equal(np.array(tensor), array)

    def test_tensor_set_from_array_list(self):
        array = np.random.randint(1000, size=(200, 300))
        list_array = [array, array]
        tensor = fluid.Tensor()
        place = core.CPUPlace()
        tensor.set(list_array, place)
        self.assertEqual([2, 200, 300], tensor.shape())
        np.testing.assert_array_equal(np.array(tensor), list_array)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            tensor.set(list_array, place)
            self.assertEqual([2, 200, 300], tensor.shape())
            np.testing.assert_array_equal(np.array(tensor), list_array)

            place = core.CUDAPinnedPlace()
            tensor.set(list_array, place)
            self.assertEqual([2, 200, 300], tensor.shape())
            np.testing.assert_array_equal(np.array(tensor), list_array)

    def test_tensor_set_error(self):
        scope = core.Scope()
        var = scope.var("test_tensor")
        place = core.CPUPlace()

        tensor = var.get_tensor()

        exception = None
        try:
            error_array = ["1", "2"]
            tensor.set(error_array, place)
        except ValueError as ex:
            exception = ex

        self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()

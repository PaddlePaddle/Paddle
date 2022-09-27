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
from paddle.fluid.lod_tensor import create_lod_tensor, create_random_int_lodtensor
import numpy as np
import unittest


class TestLoDTensor(unittest.TestCase):

    def test_pybind_recursive_seq_lens(self):
        tensor = fluid.LoDTensor()
        recursive_seq_lens = []
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        recursive_seq_lens = [[], [1], [3]]
        self.assertRaises(Exception, tensor.set_recursive_sequence_lengths,
                          recursive_seq_lens)
        recursive_seq_lens = [[0], [2], [3]]
        self.assertRaises(Exception, tensor.set_recursive_sequence_lengths,
                          recursive_seq_lens)

        recursive_seq_lens = [[1, 2, 3]]
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)
        tensor.set(np.random.random([6, 1]), fluid.CPUPlace())
        self.assertTrue(tensor.has_valid_recursive_sequence_lengths())
        tensor.set(np.random.random([9, 1]), fluid.CPUPlace())
        self.assertFalse(tensor.has_valid_recursive_sequence_lengths())

        # Each level's sum should be equal to the number of items in the next level
        # Moreover, last level's sum should be equal to the tensor height
        recursive_seq_lens = [[2, 3], [1, 3, 1, 2, 2]]
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)
        tensor.set(np.random.random([8, 1]), fluid.CPUPlace())
        self.assertFalse(tensor.has_valid_recursive_sequence_lengths())
        recursive_seq_lens = [[2, 3], [1, 3, 1, 2, 1]]
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        self.assertTrue(tensor.has_valid_recursive_sequence_lengths())
        tensor.set(np.random.random([9, 1]), fluid.CPUPlace())
        self.assertFalse(tensor.has_valid_recursive_sequence_lengths())

    def test_create_lod_tensor(self):
        # Create LoDTensor from a list
        data = [[np.int64(1), np.int64(2),
                 np.int64(3)], [np.int64(3), np.int64(4)]]
        wrong_recursive_seq_lens = [[2, 2]]
        correct_recursive_seq_lens = [[3, 2]]
        self.assertRaises(AssertionError, create_lod_tensor, data,
                          wrong_recursive_seq_lens, fluid.CPUPlace())
        tensor = create_lod_tensor(data, correct_recursive_seq_lens,
                                   fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         correct_recursive_seq_lens)
        self.assertEqual(tensor._dtype(), core.VarDesc.VarType.INT64)
        self.assertEqual(tensor.shape(), [5, 1])
        np.testing.assert_array_equal(
            np.array(tensor),
            np.array([1, 2, 3, 3, 4]).reshape(tensor.shape()).astype('int64'))

        # Create LoDTensor from numpy array
        data = np.random.random([10, 1]).astype('float64')
        recursive_seq_lens = [[2, 1], [3, 3, 4]]
        tensor = create_lod_tensor(data, recursive_seq_lens, fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)
        self.assertEqual(tensor._dtype(), core.VarDesc.VarType.FP64)
        self.assertEqual(tensor.shape(), [10, 1])
        np.testing.assert_array_equal(np.array(tensor), data)

        # Create LoDTensor from another LoDTensor, they are differnt instances
        new_recursive_seq_lens = [[2, 2, 1], [1, 2, 2, 3, 2]]
        new_tensor = create_lod_tensor(tensor, new_recursive_seq_lens,
                                       fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)
        self.assertEqual(new_tensor.recursive_sequence_lengths(),
                         new_recursive_seq_lens)

    def test_create_random_int_lodtensor(self):
        # The shape of a word, commonly used in speech and NLP problem, is [1]
        shape = [1]
        recursive_seq_lens = [[2, 3, 5]]
        dict_size = 10000
        low = 0
        high = dict_size - 1
        tensor = create_random_int_lodtensor(recursive_seq_lens, shape,
                                             fluid.CPUPlace(), low, high)
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)
        self.assertEqual(tensor.shape(), [10, 1])

    def test_print_lodtensor(self):
        shape = [1]
        recursive_seq_lens = [[2, 3, 5]]
        dict_size = 100
        low = 0
        high = dict_size - 1
        tensor = create_random_int_lodtensor(recursive_seq_lens, shape,
                                             fluid.CPUPlace(), low, high)
        print(tensor)
        self.assertTrue(isinstance(str(tensor), str))

        if core.is_compiled_with_cuda():
            gtensor = create_random_int_lodtensor(recursive_seq_lens, shape,
                                                  fluid.CUDAPlace(0), low, high)
            print(gtensor)
            self.assertTrue(isinstance(str(gtensor), str))

    def test_dlpack_support(self):
        tensor = fluid.create_lod_tensor(
            np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
            fluid.CPUPlace())
        dltensor = tensor._to_dlpack()
        tensor_from_dlpack = fluid.core.from_dlpack(dltensor)
        self.assertTrue(isinstance(tensor_from_dlpack, fluid.core.Tensor))
        np.testing.assert_array_equal(
            np.array(tensor_from_dlpack),
            np.array([[1], [2], [3], [4]]).astype('int'))
        # when build with cuda
        if core.is_compiled_with_cuda():
            gtensor = fluid.create_lod_tensor(
                np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
                fluid.CUDAPlace(0))
            gdltensor = gtensor._to_dlpack()
            gtensor_from_dlpack = fluid.core.from_dlpack(gdltensor)
            self.assertTrue(isinstance(gtensor_from_dlpack, fluid.core.Tensor))
            np.testing.assert_array_equal(
                np.array(gtensor_from_dlpack),
                np.array([[1], [2], [3], [4]]).astype('int'))

    def test_as_type(self):
        tensor = fluid.create_lod_tensor(
            np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
            fluid.CPUPlace())
        fp32_tensor = tensor._as_type(core.VarDesc.VarType.FP32)
        print(fp32_tensor)


if __name__ == '__main__':
    unittest.main()

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
        data = [[1, 2, 3], [3, 4]]
        wrong_recursive_seq_lens = [[2, 2]]
        correct_recursive_seq_lens = [[3, 2]]
        self.assertRaises(AssertionError, create_lod_tensor, data,
                          wrong_recursive_seq_lens, fluid.CPUPlace())
        tensor = create_lod_tensor(data, correct_recursive_seq_lens,
                                   fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         correct_recursive_seq_lens)

        # Create LoDTensor from numpy array
        data = np.random.random([10, 1])
        recursive_seq_lens = [[2, 1], [3, 3, 4]]
        tensor = create_lod_tensor(data, recursive_seq_lens, fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(),
                         recursive_seq_lens)

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


if __name__ == '__main__':
    unittest.main()

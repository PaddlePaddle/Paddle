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
    def test_pybind_lod(self):
        tensor = fluid.LoDTensor()

        lod = [[1, 2, 3]]
        tensor.set_recursive_sequence_lengths(lod)
        self.assertEqual(tensor.recursive_sequence_lengths(), lod)

        lod = [[2, 1], [1, 3, 1, 2, 1]]
        tensor.set_recursive_sequence_lengths(lod)
        self.assertEqual(tensor.recursive_sequence_lengths(), lod)

    def test_create_lod_tensor(self):
        # Create LoDTensor from a list
        data = [[1, 2, 3], [3, 4]]
        wrong_lod = [[2, 2]]
        correct_lod = [[3, 2]]
        self.assertRaises(AssertionError, create_lod_tensor, data, wrong_lod,
                          fluid.CPUPlace())
        tensor = create_lod_tensor(data, correct_lod, fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(), correct_lod)

        # Create LoDTensor from numpy array
        data = np.random.random([10, 1])
        lod = [[2, 1], [3, 3, 4]]
        tensor = create_lod_tensor(data, lod, fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(), lod)

        # Create LoDTensor from another LoDTensor, they are differnt instances
        new_lod = [[2, 2, 1], [1, 2, 2, 3, 2]]
        new_tensor = create_lod_tensor(tensor, new_lod, fluid.CPUPlace())
        self.assertEqual(tensor.recursive_sequence_lengths(), lod)
        self.assertEqual(new_tensor.recursive_sequence_lengths(), new_lod)

    def test_create_random_int_lodtensor(self):
        # The shape of a word, commonly used in speech and NLP problem, is [1]
        shape = [1]
        lod = [[2, 3, 5]]
        dict_size = 10000
        low = 0
        high = dict_size - 1
        tensor = create_random_int_lodtensor(lod, shape,
                                             fluid.CPUPlace(), low, high)
        self.assertEqual(tensor.recursive_sequence_lengths(), lod)
        self.assertEqual(tensor.shape(), [10, 1])


if __name__ == '__main__':
    unittest.main()

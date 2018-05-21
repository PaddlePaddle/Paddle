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

import unittest
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.lod_tensor import create_lod_tensor, create_random_int_lodtensor, _validate_lod, _convert_lod


class TestLoDTensor(unittest.TestCase):
    def test_validate_lod(self):
        lod = (1, 2, 1)
        self.assertRaises(AssertionError, _validate_lod, lod, -1)
        lod = [[1, 2], (2, 3)]
        self.assertRaises(AssertionError, _validate_lod, lod, -1)
        lod = [1, 2, 3]
        self.assertRaises(AssertionError, _validate_lod, lod, -1)

        lod = []
        self.assertTrue(_validate_lod(lod, -1))
        lod = [[], [1], [3]]
        self.assertFalse(_validate_lod(lod, -1))
        lod = [[0], [-1], [3]]
        self.assertFalse(_validate_lod(lod, -1))

        # Each level's sum should be equal to the number of items in the next level
        # Moreover, last level's sum should be equal to the tensor height
        lod = [[2, 3], [1, 3, 1, 2, 1]]
        self.assertTrue(_validate_lod(lod, tensor_height=8))
        lod = [[1, 3], [2, 1, 3]]
        self.assertFalse(_validate_lod(lod, tensor_height=6))
        lod = [[1, 3], [2, 1, 3, 4]]
        self.assertFalse(_validate_lod(lod, tensor_height=5))

    def test_convert_lod(self):
        lod = [[1, 2, 3]]
        converted_lod = [[0, 1, 3, 6]]
        self.assertEqual(_convert_lod(lod), converted_lod)

        lod = [[2, 3], [1, 3, 1, 2, 1]]
        converted_lod = [[0, 2, 5], [0, 1, 4, 5, 7, 8]]
        self.assertEqual(_convert_lod(lod), converted_lod)

    def test_create_lod_tensor(self):
        # Only numpy array or a fluid LoDTensor is valid input to
        # create_lod_tensor function, a list of lists is not.
        data = [[1, 2], [3, 4]]
        self.assertRaises(Exception, create_lod_tensor, data, [],
                          fluid.CPUPlace())

        # Create LoDTensor from numpy array
        data = numpy.random.random([10, 1])
        lod = [[2, 1], [3, 3, 4]]
        tensor = create_lod_tensor(data, lod, fluid.CPUPlace())
        self.assertEqual(tensor.lod(), [[0, 2, 3], [0, 3, 6, 10]])

        # Create LoDTensor from another LoDTensor, they are differnt instances
        new_lod = [[2, 2, 1], [1, 2, 2, 3, 2]]
        new_tensor = create_lod_tensor(tensor, new_lod, fluid.CPUPlace())
        self.assertEqual(tensor.lod(), [[0, 2, 3], [0, 3, 6, 10]])
        self.assertEqual(new_tensor.lod(), [[0, 2, 4, 5], [0, 1, 3, 5, 8, 10]])

    def test_create_random_int_lodtensor(self):
        # The shape of a word, commonly used in speech and NLP problem
        shape = [1]
        lod = [[2, 3, 5]]
        dict_size = 10000
        low = 0
        high = dict_size - 1
        tensor = create_random_int_lodtensor(lod, shape, low, high,
                                             fluid.CPUPlace())
        self.assertEqual(tensor.lod(), [[0, 2, 5, 10]])
        self.assertEqual(tensor.shape(), [10, 1])


if __name__ == '__main__':
    unittest.main()

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
from paddle.fluid.lod_tensor import create_lod_tensor, _validate_lod, _convert_lod


class TestLoDTensor(unittest.TestCase):
    def test_validate_lod_1(self):
        lod = (1, 2, 1)
        self.assertRaises(AssertionError, _validate_lod, lod, -1)
        lod = [[1, 2], [2, 3]]
        self.assertRaises(AssertionError, _validate_lod, lod, -1)

    def test_convert_lod_1(self):
        lod = [[1, 2, 3]]
        converted_lod = [[0, 1, 3, 6]]
        self.assertEqual(_convert_lod(lod), converted_lod)

    def test_create_lod_tensor_1(self):
        tensor = numpy.random.random([10, 1])
        create_lod_tensor(tensor, [[2, 2], [3, 3, 4]], fluid.CPUPlace())


if __name__ == '__main__':
    unittest.main()

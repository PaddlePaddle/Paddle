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
import paddle.fluid.core as core
import numpy


class TestLoDTensorArray(unittest.TestCase):
    def test_get_set(self):
        scope = core.Scope()
        arr = scope.var('tmp_lod_tensor_array')
        tensor_array = arr.get_lod_tensor_array()
        self.assertEqual(0, len(tensor_array))
        cpu = core.CPUPlace()
        for i in xrange(10):
            t = core.LoDTensor()
            t.set(numpy.array([i], dtype='float32'), cpu)
            t.set_recursive_sequence_lengths([[1]])
            tensor_array.append(t)

        self.assertEqual(10, len(tensor_array))

        for i in xrange(10):
            t = tensor_array[i]
            self.assertEqual(numpy.array(t), numpy.array([i], dtype='float32'))
            self.assertEqual([[1]], t.recursive_sequence_lengths())

            t = core.LoDTensor()
            t.set(numpy.array([i + 10], dtype='float32'), cpu)
            t.set_recursive_sequence_lengths([[1]])
            tensor_array[i] = t
            t = tensor_array[i]
            self.assertEqual(
                numpy.array(t), numpy.array(
                    [i + 10], dtype='float32'))
            self.assertEqual([[1]], t.recursive_sequence_lengths())


if __name__ == '__main__':
    unittest.main()

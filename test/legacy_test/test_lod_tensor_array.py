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

import numpy as np

import paddle
from paddle.base import core


class TestDenseTensorArray(unittest.TestCase):
    def test_get_set(self):
        scope = core.Scope()
        arr = scope.var('tmp_lod_tensor_array')
        tensor_array = arr.get_dense_tensor_array()
        self.assertEqual(0, len(tensor_array))
        cpu = core.CPUPlace()
        for i in range(10):
            t = core.DenseTensor()
            t.set(np.array([i], dtype='float32'), cpu)
            tensor_array.append(t)

        self.assertEqual(10, len(tensor_array))

        for i in range(10):
            t = tensor_array[i]
            self.assertEqual(np.array(t), np.array([i], dtype='float32'))

            t = core.DenseTensor()
            t.set(np.array([i + 10], dtype='float32'), cpu)
            tensor_array[i] = t
            t = tensor_array[i]
            self.assertEqual(np.array(t), np.array([i + 10], dtype='float32'))


class TestCreateArray(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CPUPlace()
        self.shapes = [[10, 4], [8, 12], [1]]

    def test_initialized_list_and_error(self):
        paddle.disable_static()
        init_data = [
            np.random.random(shape).astype('float32') for shape in self.shapes
        ]
        array = paddle.tensor.create_array(
            'float32', [paddle.to_tensor(x) for x in init_data]
        )
        for res, gt in zip(array, init_data):
            np.testing.assert_array_equal(res, gt)

        # test for None
        array = paddle.tensor.create_array('float32')
        self.assertTrue(isinstance(array, list))
        self.assertEqual(len(array), 0)

        # test error
        with self.assertRaises(TypeError):
            paddle.tensor.create_array('float32', 'str')

    def test_static(self):
        paddle.enable_static()
        init_data = [
            paddle.ones(shape, dtype='float32') for shape in self.shapes
        ]
        array = paddle.tensor.create_array('float32', init_data)
        for res, gt in zip(array, init_data):
            self.assertTrue(res.shape, gt.shape)

        # test error with nest list
        with self.assertRaises(TypeError):
            paddle.tensor.create_array(
                'float32', [init_data[0], [init_data[1]]]
            )

        # test error with not variable
        with self.assertRaises(TypeError):
            paddle.tensor.create_array('float32', ("str"))

        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid


class TestRandomCropOp(OpTest):

    def setUp(self):
        to_crop = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]] *
                           5).astype(np.int32)
        self.possible_res = [
            np.array([[1, 2, 3], [5, 6, 7]]).astype(np.int32),
            np.array([[2, 3, 4], [6, 7, 8]]).astype(np.int32),
            np.array([[5, 6, 7], [9, 10, 11]]).astype(np.int32),
            np.array([[6, 7, 8], [10, 11, 12]]).astype(np.int32)
        ]
        self.op_type = "random_crop"
        self.inputs = {'X': to_crop, 'Seed': np.array([10]).astype('int64')}
        self.outputs = {'Out': np.array([]), 'SeedOut': np.array([])}
        self.attrs = {'shape': [2, 3]}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out = np.array(outs[1])
        for ins in out[:]:
            is_equal = [(ins == res).all() for res in self.possible_res]
            self.assertIn(True, is_equal)


class TestRandomCropOpError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program()):

            def test_x_type():
                input_data = np.random.random(2, 3, 256, 256).astype("float32")
                fluid.layers.random_crop(input_data)

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype():
                x2 = fluid.layers.data(name='x2',
                                       shape=[None, 3, 256, 256],
                                       dtype='float16')
                fluid.layers.random_crop(x2)

            self.assertRaises(TypeError, test_x_dtype)

            def test_shape_type():
                x3 = fluid.layers.data(name='x3',
                                       shape=[None, 3, 256, 256],
                                       dtype='float32')
                fluid.layers.random_crop(x3, shape=1)

            self.assertRaises(TypeError, test_shape_type)


if __name__ == "__main__":
    unittest.main()

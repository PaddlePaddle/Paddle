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
from op_test import OpTest


def QuadTransform(input):
    shape = input.shape
    batch_size = shape[0]
    h = shape[2]
    w = shape[3]
    h_indexes = np.array(range(h) * w).reshape(
        [w, h]).transpose()[np.newaxis, :]  # [1, h, w]
    w_indexes = np.array(range(w) * h).reshape(
        [h, w])[np.newaxis, :]  # [1, h, w]
    indexes = np.concatenate(
        (h_indexes, w_indexes))[np.newaxis, :]  # [1, 2, h, w]
    indexes = indexes.repeat([4], axis=0)[np.newaxis, :]  # [1, 4, 2, h, w]
    indexes = indexes.repeat([batch_size], axis=0)  # [batch_size, 4, 2, h, w]
    return input + indexes.reshape(input.shape)  # [batch_size, 8, h, w]


class TestQuadTransformOp(OpTest):
    def config(self):
        self.input_shape = (1, 8, 2, 2)

    def setUp(self):
        self.op_type = "quad_transform"
        input = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'Input': input}
        output = QuadTransform(input)
        self.outputs = {'Ouput': output}

    def test_check_output(self):
        self.check_output()


class TestCase1(TestQuadTransformOp):
    def config(self):
        self.input_shape = (2, 8, 3, 2)


class TestCase2(TestQuadTransformOp):
    def config(self):
        self.input_shape = (3, 2, 4, 5)


if __name__ == '__main__':
    unittest.main()

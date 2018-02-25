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
import sys
import math
from op_test import OpTest


class TestIOUSimilarityOp(OpTest):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "iou_similarity"
        self.boxes1 = np.array(
            [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]]).astype('float32')
        self.boxes2 = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]]).astype('float32')
        self.output = np.array(
            [[2.0 / 16.0, 0, 6.0 / 400.0],
             [1.0 / 16.0, 0.0, 5.0 / 400.0]]).astype('float32')

        self.inputs = {'X': self.boxes1, 'Y': self.boxes2}

        self.outputs = {'Out': self.output}


class TestIOUSimilarityOpWithLoD(TestIOUSimilarityOp):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        super(TestIOUSimilarityOpWithLoD, self).setUp()
        self.boxes1_lod = [[0, 1, 2]]
        self.output_lod = [[0, 1, 2]]

        self.inputs = {'X': (self.boxes1, self.boxes1_lod), 'Y': self.boxes2}
        self.outputs = {'Out': (self.output, self.output_lod)}


if __name__ == '__main__':
    unittest.main()

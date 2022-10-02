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


class TestMineHardExamplesOp(OpTest):

    def set_data(self):
        self.init_test_data()
        self.inputs = {
            'ClsLoss': self.cls_loss,
            'LocLoss': self.loc_loss,
            'MatchIndices': self.match_indices,
            'MatchDist': self.match_dis
        }

        self.attrs = {
            'neg_pos_ratio': self.neg_pos_ratio,
            'neg_overlap': self.neg_overlap,
            'sample_size': self.sample_size,
            'mining_type': self.mining_type
        }

        self.outputs = {
            'NegIndices': (self.neg_indices, self.neg_indices_lod),
            'UpdatedMatchIndices': self.updated_match_indices
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        return

    def setUp(self):
        self.op_type = "mine_hard_examples"
        self.set_data()

    def init_test_data(self):
        self.neg_pos_ratio = 1.0
        self.neg_overlap = 0.5
        self.sample_size = 0
        self.mining_type = "max_negative"
        self.cls_loss = np.array([[0.1, 0.1, 0.3], [0.3, 0.1,
                                                    0.1]]).astype('float64')

        self.loc_loss = np.array([[0.1, 0.2, 0.3], [0.3, 0.4,
                                                    0.1]]).astype('float64')

        self.match_dis = np.array([[0.2, 0.4, 0.8], [0.1, 0.9,
                                                     0.3]]).astype('float64')

        self.match_indices = np.array([[0, -1, -1], [-1, 0,
                                                     -1]]).astype('int32')

        self.updated_match_indices = self.match_indices

        self.neg_indices_lod = [[1, 1]]
        self.neg_indices = np.array([[1], [0]]).astype('int32')


class TestMineHardExamplesOpHardExample(TestMineHardExamplesOp):

    def init_test_data(self):
        super(TestMineHardExamplesOpHardExample, self).init_test_data()
        self.mining_type = "hard_example"
        self.sample_size = 2

        self.cls_loss = np.array([[0.5, 0.1, 0.3], [0.3, 0.1,
                                                    0.1]]).astype('float64')

        self.loc_loss = np.array([[0.2, 0.2, 0.3], [0.3, 0.1,
                                                    0.2]]).astype('float64')

        self.match_indices = np.array([[0, -1, -1], [-1, 0,
                                                     -1]]).astype('int32')

        self.updated_match_indices = np.array([[0, -1, -1],
                                               [-1, -1, -1]]).astype('int32')

        self.neg_indices_lod = [[1, 2]]
        self.neg_indices = np.array([[2], [0], [2]]).astype('int32')


if __name__ == '__main__':
    unittest.main()

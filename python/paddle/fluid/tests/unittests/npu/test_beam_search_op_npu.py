#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import sys

sys.path.append("..")
from op_test import OpTest
import unittest
import numpy as np
import paddle.fluid as fluid

paddle.enable_static()


class TestBeamSearchNPUOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "beam_search"
        self.init_data()
        self.inputs = {
            'pre_ids': (self.pre_ids, self.lod),
            'pre_scores': (self.pre_score, self.lod),
            'ids': (self.ids, self.lod),
            'scores': (self.score, self.lod)
        }
        # The `target_lod` attribute is still based on offset
        self.attrs = {
            'level': 0,
            'beam_size': self.beam_size,
            'end_id': 0,
            'is_accumulated': self.is_accumulated
        }
        self.outputs = {
            'selected_ids': (self.selected_ids, self.out_lod),
            'selected_scores': (self.selected_scores, self.out_lod),
            'parent_idx': self.parent_idx
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_data(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.pre_ids = np.array([[1], [2], [3], [4]], dtype='int64')
        self.ids = np.array([[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]],
                            dtype='int64')
        self.lod = [[2, 2], [1, 1, 1, 1]]
        self.out_lod = [[2, 2], [1, 1, 1, 1]]
        self.offset_lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        self.score = np.array([
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.9, 0.5, 0.1],
            [0.7, 0.5, 0.1],
        ],
                              dtype='float32')
        self.pre_score = np.array([[0.1], [0.2], [0.3], [0.4]], dtype='float32')
        self.selected_ids = np.array([4, 2, 3, 8])[:, np.newaxis]
        self.selected_scores = np.array([0.5, 0.6, 0.9, 0.7])[:, np.newaxis]
        self.parent_idx = np.array([0, 1, 2, 3])

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestBeamSearchNPUOp2(TestBeamSearchNPUOp):

    def init_data(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.pre_ids = np.array([[1], [2], [3], [4]], dtype='int64')
        self.ids = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        self.lod = [[2, 2], [1, 1, 1, 1]]
        self.out_lod = [[2, 2], [2, 0, 1, 1]]
        self.offset_lod = [[0, 2, 4], [0, 2, 2, 3, 4]]
        self.score = np.array([
            [0.6, 0.9],
            [0.5, 0.3],
            [0.9, 0.5],
            [0.1, 0.7],
        ],
                              dtype='float32')
        self.pre_score = np.array([[0.1], [0.2], [0.3], [0.4]], dtype='float32')
        self.selected_ids = np.array([4, 2, 3, 1])[:, np.newaxis]
        self.selected_scores = np.array([0.6, 0.9, 0.9, 0.7])[:, np.newaxis]
        self.parent_idx = np.array([0, 0, 2, 3])


class TestBeamSearchNPUOp3(TestBeamSearchNPUOp):

    def init_data(self):
        # end_id = 0
        self.beam_size = 2
        self.is_accumulated = True
        self.pre_ids = np.array([[1], [0], [0], [4]], dtype='int64')
        self.ids = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        self.lod = [[2, 2], [1, 1, 1, 1]]
        self.out_lod = [[2, 2], [1, 1, 0, 2]]
        self.offset_lod = [[0, 2, 4], [0, 1, 2, 2, 4]]
        self.score = np.array([
            [0.6, 0.9],
            [0.5, 0.3],
            [0.9, 0.5],
            [0.6, 0.7],
        ],
                              dtype='float32')
        self.pre_score = np.array([[0.1], [1.2], [0.5], [0.4]], dtype='float32')
        self.selected_ids = np.array([2, 0, 8, 1])[:, np.newaxis]
        self.selected_scores = np.array([0.9, 1.2, 0.6, 0.7])[:, np.newaxis]
        self.parent_idx = np.array([0, 1, 3, 3])


class TestBeamSearchNPUOp4(TestBeamSearchNPUOp):

    def init_data(self):
        # is_accumulated = False
        self.beam_size = 2
        self.is_accumulated = False
        self.pre_ids = np.array([[1], [2], [3], [4]], dtype='int64')
        self.ids = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        self.lod = [[2, 2], [1, 1, 1, 1]]
        self.out_lod = [[2, 2], [0, 2, 1, 1]]
        self.offset_lod = [[0, 2, 4], [0, 0, 2, 3, 4]]
        self.score = np.array([
            [0.6, 0.9],
            [0.5, 0.3],
            [0.9, 0.5],
            [0.1, 0.7],
        ],
                              dtype='float32')
        self.pre_score = np.array([[0.1], [2.2], [0.3], [0.4]], dtype='float32')
        self.selected_ids = np.array([7, 3, 3, 1])[:, np.newaxis]
        self.selected_scores = np.array([1.50685, 0.996027, 0.194639,
                                         0.043325])[:, np.newaxis]
        self.parent_idx = np.array([1, 1, 2, 3])


class TestBeamSearchNPUOp5(TestBeamSearchNPUOp):

    def init_data(self):
        # beam_size = 1
        self.beam_size = 1
        self.is_accumulated = True
        self.pre_ids = np.array([[1], [2], [3], [4]], dtype='int64')
        self.ids = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        self.lod = [[1, 1, 1, 1], [1, 1, 1, 1]]
        self.out_lod = [[1, 1, 1, 1], [1, 1, 1, 1]]
        self.offset_lod = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        self.score = np.array([
            [0.6, 0.9],
            [0.5, 0.3],
            [0.9, 0.5],
            [0.1, 0.7],
        ],
                              dtype='float32')
        self.pre_score = np.array([[0.1], [0.2], [0.3], [0.4]], dtype='float32')
        self.selected_ids = np.array([2, 7, 3, 1])[:, np.newaxis]
        self.selected_scores = np.array([0.9, 0.5, 0.9, 0.7])[:, np.newaxis]
        self.parent_idx = np.array([0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()

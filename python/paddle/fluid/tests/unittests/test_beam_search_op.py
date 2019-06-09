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

from __future__ import print_function

import logging
from paddle.fluid.op import Operator, DynamicRecurrentOp
import paddle.fluid.core as core
import unittest
import numpy as np


def create_tensor(scope, name, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set(np_data, core.CPUPlace())
    return tensor


class BeamSearchOpTester(unittest.TestCase):
    """unittest of beam_search_op"""

    def setUp(self):
        self.scope = core.Scope()
        self._create_ids()
        self._create_pre_scores()
        self._create_scores()
        self._create_pre_ids()
        self.scope.var('selected_ids')
        self.scope.var('selected_scores')
        self.scope.var('parent_idx')

    def test_run(self):
        op = Operator(
            'beam_search',
            pre_ids='pre_ids',
            pre_scores='pre_scores',
            ids='ids',
            scores='scores',
            selected_ids='selected_ids',
            selected_scores='selected_scores',
            parent_idx='parent_idx',
            level=0,
            beam_size=2,
            end_id=0, )
        op.run(self.scope, core.CPUPlace())
        selected_ids = self.scope.find_var("selected_ids").get_tensor()
        selected_scores = self.scope.find_var("selected_scores").get_tensor()
        parent_idx = self.scope.find_var("parent_idx").get_tensor()
        self.assertTrue(
            np.allclose(
                np.array(selected_ids), np.array([4, 2, 3, 8])[:, np.newaxis]))
        self.assertTrue(
            np.allclose(
                np.array(selected_scores),
                np.array([0.5, 0.6, 0.9, 0.7])[:, np.newaxis]))
        self.assertEqual(selected_ids.lod(), [[0, 2, 4], [0, 1, 2, 3, 4]])
        self.assertTrue(
            np.allclose(np.array(parent_idx), np.array([0, 1, 2, 3])))

    def _create_pre_ids(self):
        np_data = np.array([[1, 2, 3, 4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array(
            [[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.9, 0.5, 0.1],
                [0.7, 0.5, 0.1],
            ],
            dtype='float32')
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)


if __name__ == '__main__':
    unittest.main()

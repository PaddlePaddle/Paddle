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
from op import Operator

from paddle.base import core


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
        self.set_outputs()
        self.scope.var('selected_ids').get_tensor()
        self.scope.var('selected_scores').get_tensor()
        self.scope.var('parent_idx').get_tensor()

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
            beam_size=self.beam_size,
            end_id=0,
            is_accumulated=self.is_accumulated,
        )
        op.run(self.scope, core.CPUPlace())
        selected_ids = self.scope.find_var("selected_ids").get_tensor()
        selected_scores = self.scope.find_var("selected_scores").get_tensor()
        parent_idx = self.scope.find_var("parent_idx").get_tensor()
        np.testing.assert_allclose(
            np.array(selected_ids), self.output_ids, rtol=1e-05
        )
        np.testing.assert_allclose(
            np.array(selected_scores), self.output_scores, rtol=1e-05
        )
        self.assertEqual(selected_ids.lod(), self.output_lod)
        np.testing.assert_allclose(
            np.array(parent_idx), self.output_parent_idx, rtol=1e-05
        )

    def _create_pre_ids(self):
        np_data = np.array([[1, 2, 3, 4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array(
            [[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]], dtype='int64'
        )
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
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.output_ids = np.array([4, 2, 3, 8])[:, np.newaxis]
        self.output_scores = np.array([0.5, 0.6, 0.9, 0.7])[:, np.newaxis]
        self.output_lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        self.output_parent_idx = np.array([0, 1, 2, 3])


class BeamSearchOpTester2(BeamSearchOpTester):
    def _create_pre_ids(self):
        np_data = np.array([[1], [2], [3], [4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.6, 0.9],
                [0.5, 0.3],
                [0.9, 0.5],
                [0.1, 0.7],
            ],
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.output_ids = np.array([2, 4, 3, 1])[:, np.newaxis]
        self.output_scores = np.array([0.9, 0.6, 0.9, 0.7])[:, np.newaxis]
        self.output_lod = [[0, 2, 4], [0, 2, 2, 3, 4]]
        self.output_parent_idx = np.array([0, 0, 2, 3])


class BeamSearchOpTester3(BeamSearchOpTester):
    # pre_id = end_id
    def _create_pre_ids(self):
        np_data = np.array([[1], [0], [0], [4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1], [1.2], [0.5], [0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.6, 0.9],
                [0.5, 0.3],
                [0.9, 0.5],
                [0.6, 0.7],
            ],
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.output_ids = np.array([2, 0, 1, 8])[:, np.newaxis]
        self.output_scores = np.array([0.9, 1.2, 0.7, 0.6])[:, np.newaxis]
        self.output_lod = [[0, 2, 4], [0, 1, 2, 2, 4]]
        self.output_parent_idx = np.array([0, 1, 3, 3])


class BeamSearchOpTester4(BeamSearchOpTester):
    # prune beam search while pre_id of in all beams is end_id
    def _create_pre_ids(self):
        np_data = np.array([[0], [0], [0], [4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1], [1.2], [0.5], [0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.6, 0.9],
                [0.5, 0.3],
                [0.9, 0.5],
                [0.6, 0.7],
            ],
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 2
        self.is_accumulated = True
        self.output_ids = np.array([1, 8])[:, np.newaxis]
        self.output_scores = np.array([0.7, 0.6])[:, np.newaxis]
        self.output_lod = [[0, 2, 4], [0, 0, 0, 0, 2]]
        self.output_parent_idx = np.array([3, 3])


class BeamSearchOpTester5(BeamSearchOpTester):
    # is_accumulated = False
    def _create_pre_ids(self):
        np_data = np.array([[1], [2], [3], [4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1, 2.2, 0.3, 0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 2, 4], [0, 1, 2, 3, 4]]
        np_data = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.6, 0.9],
                [0.5, 0.3],
                [0.9, 0.5],
                [0.1, 0.7],
            ],
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 2
        self.is_accumulated = False
        self.output_ids = np.array([7, 3, 3, 1])[:, np.newaxis]
        self.output_scores = np.array([1.50685, 0.996027, 0.194639, 0.043325])[
            :, np.newaxis
        ]
        self.output_lod = [[0, 2, 4], [0, 0, 2, 3, 4]]
        self.output_parent_idx = np.array([1, 1, 2, 3])


class BeamSearchOpTester6(BeamSearchOpTester):
    # beam_size = 1
    def _create_pre_ids(self):
        np_data = np.array([[1], [2], [3], [4]], dtype='int64')
        tensor = create_tensor(self.scope, 'pre_ids', np_data)

    def _create_pre_scores(self):
        np_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='float32')
        tensor = create_tensor(self.scope, 'pre_scores', np_data)

    def _create_ids(self):
        self.lod = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        np_data = np.array([[4, 2], [7, 3], [3, 5], [8, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.6, 0.9],
                [0.5, 0.3],
                [0.9, 0.5],
                [0.1, 0.7],
            ],
            dtype='float32',
        )
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)

    def set_outputs(self):
        self.beam_size = 1
        self.is_accumulated = True
        self.output_ids = np.array([2, 7, 3, 1])[:, np.newaxis]
        self.output_scores = np.array([0.9, 0.5, 0.9, 0.7])[:, np.newaxis]
        self.output_lod = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        self.output_parent_idx = np.array([0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()

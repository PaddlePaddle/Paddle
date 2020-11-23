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
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard


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


class TestBeamSearchOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            pre_ids = fluid.data(
                name='pre_id', shape=[1], lod_level=2, dtype='int64')
            pre_scores = fluid.data(
                name='pre_scores', shape=[1], lod_level=2, dtype='float32')
            probs = fluid.data(name='probs', shape=[10000], dtype='float32')
            topk_scores, topk_indices = fluid.layers.topk(probs, k=4)
            accu_scores = fluid.layers.elementwise_add(
                x=fluid.layers.log(x=topk_scores),
                y=fluid.layers.reshape(
                    pre_scores, shape=[-1]),
                axis=0)

            def test_preids_Variable():
                # the input pre_ids must be Variable
                preids_data = np.random.randint(1, 5, [5, 1]).astype("int64")
                fluid.layers.beam_search(
                    pre_ids=preids_data,
                    pre_scores=pre_scores,
                    ids=topk_indices,
                    scores=accu_scores,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_preids_Variable)

            def test_prescores_Variable():
                # the input pre_scores must be Variable
                prescores_data = np.random.uniform(1, 5,
                                                   [5, 1]).astype("float32")
                fluid.layers.beam_search(
                    pre_ids=pre_ids,
                    pre_scores=prescores_data,
                    ids=topk_indices,
                    scores=accu_scores,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_prescores_Variable)

            def test_ids_Variable():
                # the input ids must be Variable or None
                ids_data = np.random.randint(1, 5, [5, 1]).astype("int64")
                fluid.layers.beam_search(
                    pre_ids=pre_ids,
                    pre_scores=pre_scores,
                    ids=ids_data,
                    scores=accu_scores,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_ids_Variable)

            def test_scores_Variable():
                # the input scores must be Variable
                scores_data = np.random.uniform(1, 5, [5, 1]).astype("float32")
                fluid.layers.beam_search(
                    pre_ids=pre_ids,
                    pre_scores=pre_scores,
                    ids=topk_indices,
                    scores=scores_data,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_scores_Variable)

            def test_preids_dtype():
                # the dtype of input pre_ids must be int64
                preids_type_data = fluid.data(
                    name='preids_type_data',
                    shape=[1],
                    lod_level=2,
                    dtype='float32')
                fluid.layers.beam_search(
                    pre_ids=preids_type_data,
                    pre_scores=pre_scores,
                    ids=topk_indices,
                    scores=accu_scores,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_preids_dtype)

            def test_prescores_dtype():
                # the dtype of input pre_scores must be float32
                prescores_type_data = fluid.data(
                    name='prescores_type_data',
                    shape=[1],
                    lod_level=2,
                    dtype='int64')
                fluid.layers.beam_search(
                    pre_ids=pre_ids,
                    pre_scores=prescores_type_data,
                    ids=topk_indices,
                    scores=accu_scores,
                    beam_size=4,
                    end_id=1)

            self.assertRaises(TypeError, test_prescores_dtype)


if __name__ == '__main__':
    unittest.main()

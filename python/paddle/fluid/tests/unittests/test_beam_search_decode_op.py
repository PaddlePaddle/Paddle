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

import unittest

import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestBeamSearchDecodeOp(unittest.TestCase):
    """unittest of beam_search_decode_op"""

    def setUp(self):
        self.scope = core.Scope()
        self.place = core.CPUPlace()

    def append_lod_tensor(self, tensor_array, lod, data):
        lod_tensor = core.LoDTensor()
        lod_tensor.set_lod(lod)
        lod_tensor.set(data, self.place)
        tensor_array.append(lod_tensor)

    def test_get_set(self):
        ids = self.scope.var("ids").get_lod_tensor_array()
        scores = self.scope.var("scores").get_lod_tensor_array()
        # Construct sample data with 5 steps and 2 source sentences
        # beam_size = 2, end_id = 1
        # start with start_id
        [
            self.append_lod_tensor(
                array, [[0, 1, 2], [0, 1, 2]], np.array(
                    [0, 0], dtype=dtype))
            for array, dtype in ((ids, "int64"), (scores, "float32"))
        ]
        [
            self.append_lod_tensor(
                array, [[0, 1, 2], [0, 2, 4]],
                np.array(
                    [2, 3, 4, 5], dtype=dtype))
            for array, dtype in ((ids, "int64"), (scores, "float32"))
        ]
        [
            self.append_lod_tensor(
                array, [[0, 2, 4], [0, 2, 2, 4, 4]],
                np.array(
                    [3, 1, 5, 4], dtype=dtype))
            for array, dtype in ((ids, "int64"), (scores, "float32"))
        ]
        [
            self.append_lod_tensor(
                array, [[0, 2, 4], [0, 1, 2, 3, 4]],
                np.array(
                    [1, 1, 3, 5], dtype=dtype))
            for array, dtype in ((ids, "int64"), (scores, "float32"))
        ]
        [
            self.append_lod_tensor(
                array, [[0, 2, 4], [0, 0, 0, 2, 2]],
                np.array(
                    [5, 1], dtype=dtype))
            for array, dtype in ((ids, "int64"), (scores, "float32"))
        ]

        sentence_ids = self.scope.var("sentence_ids").get_tensor()
        sentence_scores = self.scope.var("sentence_scores").get_tensor()

        beam_search_decode_op = Operator(
            "beam_search_decode",
            # inputs
            Ids="ids",
            Scores="scores",
            # outputs
            SentenceIds="sentence_ids",
            SentenceScores="sentence_scores",
            beam_size=2,
            end_id=1, )

        beam_search_decode_op.run(self.scope, self.place)

        expected_lod = [[0, 2, 4], [0, 4, 7, 12, 17]]
        self.assertEqual(sentence_ids.lod(), expected_lod)
        self.assertEqual(sentence_scores.lod(), expected_lod)

        expected_data = np.array(
            [0, 2, 3, 1, 0, 2, 1, 0, 4, 5, 3, 5, 0, 4, 5, 3, 1], "int64")
        self.assertTrue(np.array_equal(np.array(sentence_ids), expected_data))
        self.assertTrue(
            np.array_equal(np.array(sentence_scores), expected_data))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestBeamSearchDecodeOpGPU(TestBeamSearchDecodeOp):
    def setUp(self):
        self.scope = core.Scope()
        self.place = core.CUDAPlace(0)


if __name__ == '__main__':
    unittest.main()

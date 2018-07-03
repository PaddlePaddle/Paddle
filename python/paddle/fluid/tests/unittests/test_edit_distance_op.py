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


def Levenshtein(hyp, ref):
    """ Compute the Levenshtein distance between two strings.

    :param hyp: hypothesis string in index
    :type hyp: list
    :param ref: reference string in index
    :type ref: list
    """
    m = len(hyp)
    n = len(ref)
    if m == 0:
        return n
    if n == 0:
        return m

    dist = np.zeros((m + 1, n + 1)).astype("float32")
    for i in range(0, m + 1):
        dist[i][0] = i
    for j in range(0, n + 1):
        dist[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            deletion = dist[i - 1][j] + 1
            insertion = dist[i][j - 1] + 1
            substitution = dist[i - 1][j - 1] + cost
            dist[i][j] = min(deletion, insertion, substitution)
    return dist[m][n]


class TestEditDistanceOp(OpTest):
    def setUp(self):
        self.op_type = "edit_distance"
        normalized = False
        x1 = np.array([[12, 3, 5, 8, 2]]).astype("int64")
        x2 = np.array([[12, 4, 7, 8]]).astype("int64")
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        x1_lod = [1, 4]
        x2_lod = [3, 1]

        num_strs = len(x1_lod)
        distance = np.zeros((num_strs, 1)).astype("float32")
        sequence_num = np.array(2).astype("int64")

        x1_offset = 0
        x2_offset = 0
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=x1[x1_offset:(x1_offset + x1_lod[i])],
                ref=x2[x2_offset:(x2_offset + x2_lod[i])])
            x1_offset += x1_lod[i]
            x2_offset += x2_lod[i]
            if normalized is True:
                len_ref = x2_lod[i]
                distance[i] = distance[i] / len_ref

        self.attrs = {'normalized': normalized}
        self.inputs = {'Hyps': (x1, [x1_lod]), 'Refs': (x2, [x2_lod])}
        self.outputs = {'Out': distance, 'SequenceNum': sequence_num}

    def test_check_output(self):
        self.check_output()


class TestEditDistanceOpNormalized(OpTest):
    def setUp(self):
        self.op_type = "edit_distance"
        normalized = True
        x1 = np.array([[10, 3, 6, 5, 8, 2]]).astype("int64")
        x2 = np.array([[10, 4, 6, 7, 8]]).astype("int64")
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        x1_lod = [1, 2, 3]
        x2_lod = [2, 1, 2]

        num_strs = len(x1_lod)
        distance = np.zeros((num_strs, 1)).astype("float32")
        sequence_num = np.array(3).astype("int64")

        x1_offset = 0
        x2_offset = 0
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=x1[x1_offset:(x1_offset + x1_lod[i])],
                ref=x2[x2_offset:(x2_offset + x2_lod[i])])
            x1_offset += x1_lod[i]
            x2_offset += x2_lod[i]
            if normalized is True:
                len_ref = x2_lod[i]
                distance[i] = distance[i] / len_ref

        self.attrs = {'normalized': normalized}
        self.inputs = {'Hyps': (x1, [x1_lod]), 'Refs': (x2, [x2_lod])}
        self.outputs = {'Out': distance, 'SequenceNum': sequence_num}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()

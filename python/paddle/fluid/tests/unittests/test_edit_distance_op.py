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
import paddle


def python_edit_distance(input,
                         label,
                         input_length=None,
                         label_length=None,
                         normalized=True,
                         ignored_tokens=None):
    return paddle.nn.functional.loss.edit_distance(
        input,
        label,
        normalized=normalized,
        ignored_tokens=ignored_tokens,
        input_length=input_length,
        label_length=label_length)


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
        self.python_api = python_edit_distance
        normalized = False
        x1 = np.array([[12, 3, 5, 8, 2]]).astype("int64")
        x2 = np.array([[12, 4, 7, 8]]).astype("int64")
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        self.x1_lod = [1, 4]
        self.x2_lod = [3, 1]

        num_strs = len(self.x1_lod)
        distance = np.zeros((num_strs, 1)).astype("float32")
        sequence_num = np.array(2).astype("int64")

        x1_offset = 0
        x2_offset = 0
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=x1[x1_offset:(x1_offset + self.x1_lod[i])],
                ref=x2[x2_offset:(x2_offset + self.x2_lod[i])])
            x1_offset += self.x1_lod[i]
            x2_offset += self.x2_lod[i]
            if normalized is True:
                len_ref = self.x2_lod[i]
                distance[i] = distance[i] / len_ref

        self.attrs = {'normalized': normalized}
        self.inputs = {'Hyps': (x1, [self.x1_lod]), 'Refs': (x2, [self.x2_lod])}
        self.outputs = {'Out': distance, 'SequenceNum': sequence_num}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestEditDistanceOpNormalizedCase0(OpTest):

    def reset_config(self):
        pass

    def post_config(self):
        pass

    def setUp(self):
        self.op_type = "edit_distance"
        self.python_api = python_edit_distance
        normalized = True
        self.x1 = np.array([[10, 3, 6, 5, 8, 2]]).astype("int64")
        self.x2 = np.array([[10, 4, 6, 7, 8]]).astype("int64")
        self.x1_lod = [3, 0, 3]
        self.x2_lod = [2, 1, 2]
        self.x1 = np.transpose(self.x1)
        self.x2 = np.transpose(self.x2)

        self.reset_config()

        num_strs = len(self.x1_lod)
        distance = np.zeros((num_strs, 1)).astype("float32")
        sequence_num = np.array(num_strs).astype("int64")

        x1_offset = 0
        x2_offset = 0
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=self.x1[x1_offset:(x1_offset + self.x1_lod[i])],
                ref=self.x2[x2_offset:(x2_offset + self.x2_lod[i])])
            x1_offset += self.x1_lod[i]
            x2_offset += self.x2_lod[i]
            if normalized is True:
                len_ref = self.x2_lod[i]
                distance[i] = distance[i] / len_ref

        self.attrs = {'normalized': normalized}
        self.inputs = {
            'Hyps': (self.x1, [self.x1_lod]),
            'Refs': (self.x2, [self.x2_lod])
        }
        self.outputs = {'Out': distance, 'SequenceNum': sequence_num}

        self.post_config()

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestEditDistanceOpNormalizedCase1(TestEditDistanceOpNormalizedCase0):

    def reset_config(self):
        self.x1_lod = [0, 6, 0]
        self.x2_lod = [2, 1, 2]


class TestEditDistanceOpNormalizedCase2(TestEditDistanceOpNormalizedCase0):

    def reset_config(self):
        self.x1_lod = [0, 0, 6]
        self.x2_lod = [2, 2, 1]


class TestEditDistanceOpNormalizedTensor(OpTest):

    def reset_config(self):
        self.x1 = np.array([[10, 3, 0, 0], [6, 5, 8, 2]], dtype=np.int64)
        self.x2 = np.array([[10, 4, 0], [6, 7, 8]], dtype=np.int64)
        self.x1_lod = np.array([2, 4], dtype=np.int64)
        self.x2_lod = np.array([2, 3], dtype=np.int64)

    def setUp(self):
        self.op_type = "edit_distance"
        self.python_api = python_edit_distance
        normalized = True

        self.reset_config()

        num_strs = len(self.x1_lod)
        distance = np.zeros((num_strs, 1)).astype("float32")
        sequence_num = np.array(num_strs).astype("int64")

        for i in range(0, num_strs):
            distance[i] = Levenshtein(hyp=self.x1[i][0:self.x1_lod[i]],
                                      ref=self.x2[i][0:self.x2_lod[i]])
            if normalized is True:
                len_ref = self.x2_lod[i]
                distance[i] = distance[i] / len_ref

        self.attrs = {'normalized': normalized}
        self.inputs = {
            'Hyps': self.x1,
            'Refs': self.x2,
            'HypsLength': self.x1_lod,
            'RefsLength': self.x2_lod
        }
        self.outputs = {'Out': distance, 'SequenceNum': sequence_num}

    def test_check_output(self):
        self.check_output(check_eager=True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

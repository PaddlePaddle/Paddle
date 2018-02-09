#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import unittest
import numpy as np
from op_test import OpTest


def bipartite_match(distance, match_indices, match_dist):
    """Bipartite Matching algorithm.
    Arg:
        distance (numpy.array) : The distance of two entries with shape [M, N].
        match_indices (numpy.array): the matched indices from column to row
            with shape [1, N], it must be initialized to -1.
        match_dist (numpy.array): The matched distance from column to row
            with shape [1, N], it must be initialized to 0.
    """
    match_pair = []
    row, col = distance.shape
    for i in range(row):
        for j in range(col):
            match_pair.append((i, j, distance[i][j]))

    match_sorted = sorted(match_pair, key=lambda tup: tup[2], reverse=True)

    row_indices = -1 * np.ones((row, ), dtype=np.int)

    idx = 0
    for i, j, dist in match_sorted:
        if idx >= row:
            break
        if match_indices[j] == -1 and row_indices[i] == -1 and dist > 0:
            match_indices[j] = i
            row_indices[i] = j
            match_dist[j] = dist
            idx += 1


def batch_bipartite_match(distance, lod):
    """Bipartite Matching algorithm for batch input.
    Arg:
        distance (numpy.array) : The distance of two entries with shape [M, N].
        lod (list of int): The offsets of each input in this batch.
    """
    n = len(lod) - 1
    m = distance.shape[1]
    match_indices = -1 * np.ones((n, m), dtype=np.int)
    match_dist = np.zeros((n, m), dtype=np.float32)
    for i in range(len(lod) - 1):
        bipartite_match(distance[lod[i]:lod[i + 1], :], match_indices[i, :],
                        match_dist[i, :])
    return match_indices, match_dist


class TestBipartiteMatchOpWithLoD(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[0, 5, 11, 23]]
        dist = np.random.random((23, 217)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {
            'ColToRowMatchIndices': (match_indices),
            'ColToRowMatchDist': (match_dist),
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWithoutLoD(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[0, 8]]
        dist = np.random.random((8, 17)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': dist}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()

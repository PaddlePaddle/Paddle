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
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
import paddle.compat as cpt

class TestSequencePyramidEmbeddingOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_pyramid_embedding"
        self.emb_size = 2
        np.random.seed(1)
        table = np.random.random((17 * self.emb_size + self.emb_size, 1)).astype("float32")
        self.in_seq = np.random.randint(0, 10, (7, 1)).astype("int64")
        print(self.in_seq)
        print(table)
        self.lod = [[4, 3]]
        self.min_win_size = 2
        self.max_win_size = 3
        self.num_hash = 2
        self.attrs = {'is_sparse': True, 'rand_len': self.emb_size, 
                      'min_win_size': self.min_win_size, 'max_win_size': self.max_win_size,
                      'num_hash':self.num_hash, 'mod_by':17 * self.emb_size}
        self.inputs = {'W': table, 'Ids': (self.in_seq, self.lod)}
        hash_ids = [[24, 11],
                    [15,  0],
                    [13,  2],
                    [ 4,  0],
                    [ 5, 14],
                    [ 1,  1],
                    [12,  8],
                    [32,  6]]
        out_idx = np.reshape(np.array([[[i, i + 1] for i in x ] for x in hash_ids]), [-1, 2 * self.emb_size])
        self.outputs = {
            'Out': np.reshape(table[out_idx], [-1, 2 * self.emb_size]),
            'HashIds':np.array(hash_ids)
        }

    def test_check_output(self):
        self.check_output()

    # not surpported
    #def test_check_grad(self):
    #    self.check_grad(['W'], ['Out'], max_relative_error=0.005)

class TestSequencePyramidEmbeddingOpBigWin(TestSequencePyramidEmbeddingOp):
    def setUp(self):
        self.op_type = "sequence_pyramid_embedding"
        self.emb_size = 2
        np.random.seed(1)
        table = np.random.random((17 * self.emb_size + self.emb_size, 1)).astype("float32")
        self.in_seq = np.random.randint(0, 10, (7, 1)).astype("int64")
        print(self.in_seq)
        print(table)
        self.lod = [[4, 3]]
        self.min_win_size = 20
        self.max_win_size = 20
        self.num_hash = 2
        self.attrs = {'is_sparse': True, 'rand_len': self.emb_size, 
                      'min_win_size': self.min_win_size, 'max_win_size': self.max_win_size,
                      'num_hash':self.num_hash, 'mod_by':17 * self.emb_size}
        self.inputs = {'W': table, 'Ids': (self.in_seq, self.lod)}
        hash_ids = [[]]
        out_idx = np.reshape(np.array([[[i, i + 1] for i in x ] for x in hash_ids]), [-1, 2 * self.emb_size])
        self.outputs = {
            'Out': np.reshape(np.array([]), [-1, 2 * self.emb_size]),
            'HashIds':np.reshape(np.array(hash_ids), [-1, 2])
        }

if __name__ == "__main__":
    unittest.main()

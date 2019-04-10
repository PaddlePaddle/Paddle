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


class TestFusedEmbeddingSeqPoolOp(OpTest):
    def setUp(self):
        self.op_type = "fused_embedding_seq_pool"
        self.emb_size = 2
        table = np.random.random((17, self.emb_size)).astype("float32")
        ids = np.array([[[4], [3]], [[4], [3]], [[2], [1]],
                        [[16], [1]]]).astype("int64")
        merged_ids = np.array([4, 2, 16]).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.lod = [[3, 1]]
        self.attrs = {'is_sparse': True}
        self.inputs = {'W': table, 'Ids': (ids_expand, self.lod)}
        self.outputs = {
            'Out': np.reshape(
                np.array([
                    table[[4, 3]] + table[[4, 3]] + table[[2, 1]],
                    table[[16, 1]]
                ]), [len(self.lod[0]), 2 * self.emb_size])
        }

    def test_check_output(self):
        # TODO(minqiyang): support fusion op in dygraph mode
        self.check_output(check_dygraph=False)


if __name__ == "__main__":
    unittest.main()

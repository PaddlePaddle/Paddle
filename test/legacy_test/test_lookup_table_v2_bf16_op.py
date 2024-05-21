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

import unittest

import numpy as np
import test_lookup_table_bf16_op

import paddle
from paddle.base import core


class TestLookupTableV2BF16Op(test_lookup_table_bf16_op.TestLookupTableBF16Op):
    def init_test(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = 4
        self.mkldnn_data_type = "bfloat16"


class TestLookupTableV2BF16OpIds4D(
    test_lookup_table_bf16_op.TestLookupTableBF16OpIds4D
):
    def init_test(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = (2, 4, 5)
        self.mkldnn_data_type = "bfloat16"


class TestLookupTableV2BF16OpWIsSelectedRows(
    test_lookup_table_bf16_op.TestLookupTableBF16OpWIsSelectedRows
):
    def init_test(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = 10


class TestLookupTableV2BF16OpWIsSelectedRows4DIds(
    test_lookup_table_bf16_op.TestLookupTableBF16OpWIsSelectedRows4DIds
):
    def init_test(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = (3, 4, 5)


class TestLookupTableBF16OpWithPadding(TestLookupTableV2BF16Op):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace())


class TestLookupTableBF16OpIds4DPadding(TestLookupTableV2BF16OpIds4D):
    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace())


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

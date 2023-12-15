#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle
import paddle.nn.functional as F


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestLookupTableOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.table_np = self.random(
            shape=self.case["table_shape"], dtype=self.case["table_dtype"]
        )
        self.ids_np = self.random(
            shape=self.case["ids_shape"],
            dtype=self.case["ids_dtype"],
            low=0,
            high=self.case["table_shape"][0],
        )

    def build_paddle_program(self, target):
        table = paddle.to_tensor(self.table_np, stop_gradient=False)
        ids = paddle.to_tensor(self.ids_np, stop_gradient=False)
        out = F.embedding(ids, table, self.case["padding_idx"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("lookup_table")
        table = builder.create_input(
            self.nptype2cinntype(self.table_np.dtype),
            self.table_np.shape,
            "table",
        )
        ids = builder.create_input(
            self.nptype2cinntype(self.ids_np.dtype),
            self.ids_np.shape + (1,),
            "ids",
        )
        out = builder.lookup_table(table, ids, self.case["padding_idx"])
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [table, ids], [self.table_np, self.ids_np], [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestLookupTableOpAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLookupTableOpCase"
        self.cls = TestLookupTableOp
        self.inputs = [
            {
                "table_shape": [128, 8],
                "ids_shape": [8],
            },
            {
                "table_shape": [256, 4],
                "ids_shape": [8, 4],
            },
            {
                "table_shape": [1024, 2],
                "ids_shape": [8, 4, 2],
            },
        ]
        self.dtypes = [
            {
                "table_dtype": "float16",
                "ids_dtype": "int16",
            },
            {
                "table_dtype": "float32",
                "ids_dtype": "int32",
            },
            {
                "table_dtype": "float64",
                "ids_dtype": "int64",
            },
        ]
        self.attrs = [
            {
                "padding_idx": -1,
            },
            {
                "padding_idx": 0,
            },
            {
                "padding_idx": 1,
            },
        ]


if __name__ == "__main__":
    TestLookupTableOpAll().run()

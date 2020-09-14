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

import paddle.fluid as fluid
import paddle.fluid.core as core


class TestLookupTableFuseOp(unittest.TestCase):
    def test_fuse(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = fluid.global_scope()
        init_program = fluid.Program()

        lr = scope.var("LearningRate")
        lr.get_tensor().set([0.01], place)

        ids = [i for i in range(100)]
        out = scope.var("output")

        metas = []
        metas.append(
            "embedding_1.block0:Param,Moment1,Moment2:8,8,8:0:embedding_1@GRAD.block0:embedding_1.block0,embedding_1_moment1_0,embedding_1_moment2_0,kSparseIDs@embedding_1.block0:uniform_random&0&-0.5&0.5,fill_constant&0.0,fill_constant&0.0:none"
        )
        metas.append(
            "embedding_2.block0:Param:8:0:embedding_2@GRAD.block0:embedding_2.block0,kSparseIDs@embedding_2.block0:uniform_random&0&-0.5&0.5:none"
        )

        init_program.global_block().append_op(
            type="lookup_sparse_table_init",
            inputs=None,
            outputs=None,
            attrs={"large_scale_metas": metas})

        init_program.global_block().append_op(
            type="lookup_sparse_table_read",
            inputs={"Ids": ids},
            outputs={"Out": out},
            attrs={
                "tablename": "embedding_1.block0",
                "init": True,
                "value_names": ["Param", "Moment1", "Moment2"],
            })

        init_program.global_block().append_op(
            type="lookup_sparse_table_read",
            inputs={"Ids": ids},
            outputs={"Out": out},
            attrs={
                "tablename": "embedding_2.block0",
                "init": True,
                "value_names": ["Param"],
            })

        executor = fluid.Executor(fluid.CPUPlace())
        executor.run(init_program)

        training_program = fluid.Program()

        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 7

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        training_program.global_block().append_op(
            type="lookup_sparse_table_fuse_adam",
            inputs={"Grad": ids,
                    "LearningRate": lr},
            outputs={"Out": out},
            attrs={
                "is_entry": False,
                "tablename": "embedding_1.block0",
                "value_names": ["Param", "Moment1", "Moment2"],
            })

        training_program.global_block().append_op(
            type="lookup_sparse_table_fuse_sgd",
            inputs={"Grad": ids,
                    "LearningRate": lr},
            outputs={"Out": out},
            attrs={
                "is_entry": False,
                "tablename": "embedding_2.block0",
                "value_names": ["Param"],
            })


if __name__ == "__main__":
    unittest.main()

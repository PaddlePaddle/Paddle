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

import paddle

paddle.enable_static()

import unittest
import paddle.fluid as fluid
from paddle.fluid.framework import default_main_program
from paddle.fluid.entry_attr import ProbabilityEntry, CountFilterEntry


class EntryAttrChecks(unittest.TestCase):

    def embedding_layer(self):
        prog = fluid.Program()
        scope = fluid.core.Scope()

        with fluid.scope_guard(scope):
            with fluid.program_guard(prog):
                input = fluid.layers.data(name="dnn_data",
                                          shape=[-1, 1],
                                          dtype="int64",
                                          lod_level=1,
                                          append_batch_size=False)
                emb = fluid.layers.embedding(
                    input=input,
                    size=[100, 10],
                    is_sparse=True,
                    is_distributed=True,
                    param_attr=fluid.ParamAttr(name="deep_embedding"))
                pool = fluid.layers.sequence_pool(input=emb, pool_type="sum")
                predict = fluid.layers.fc(input=pool, size=2, act='softmax')

        block = prog.global_block()
        for op in block.ops:
            if op.type == "lookup_table":
                is_sparse = op.attr("is_sparse")
                is_distributed = op.attr("is_distributed")

                self.assertFalse(is_distributed)
                self.assertTrue(is_sparse)


class TestEntryAttrs(EntryAttrChecks):

    def test_embedding_layer(self):
        self.embedding_layer()


if __name__ == '__main__':
    unittest.main()

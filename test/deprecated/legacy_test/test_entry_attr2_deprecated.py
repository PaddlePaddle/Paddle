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

from paddle import base


class EntryAttrChecks(unittest.TestCase):
    def embedding_layer(self):
        prog = base.Program()
        scope = base.core.Scope()

        with base.scope_guard(scope):
            with base.program_guard(prog):
                input = paddle.static.data(
                    name="dnn_data", shape=[-1, 1], dtype="int64"
                )
                emb = paddle.static.nn.embedding(
                    input=input,
                    size=[100, 10],
                    is_sparse=True,
                    is_distributed=True,
                    param_attr=base.ParamAttr(name="deep_embedding"),
                )

                pool = paddle.static.nn.sequence_lod.sequence_pool(
                    input=emb, pool_type="sum"
                )
                predict = paddle.static.nn.fc(
                    x=pool, size=2, activation='softmax'
                )

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

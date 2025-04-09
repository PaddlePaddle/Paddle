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
from paddle.distributed import (
    CountFilterEntry,
    ProbabilityEntry,
    ShowClickEntry,
)


class EntryAttrChecks(unittest.TestCase):
    def base(self):
        with self.assertRaises(NotImplementedError):
            from paddle.distributed.entry_attr import EntryAttr

            base = EntryAttr()
            base._to_attr()

    def probability_entry(self):
        prob = ProbabilityEntry(0.5)
        ss = prob._to_attr()
        self.assertEqual("probability_entry:0.5", ss)

        with self.assertRaises(ValueError):
            prob1 = ProbabilityEntry("none")

        with self.assertRaises(ValueError):
            prob2 = ProbabilityEntry(-1)

    def countfilter_entry(self):
        counter = CountFilterEntry(20)
        ss = counter._to_attr()
        self.assertEqual("count_filter_entry:20", ss)

        with self.assertRaises(ValueError):
            counter1 = CountFilterEntry("none")

        with self.assertRaises(ValueError):
            counter2 = CountFilterEntry(-1)

    def showclick_entry(self):
        showclick = ShowClickEntry("show", "click")
        ss = showclick._to_attr()
        self.assertEqual("show_click_entry:show:click", ss)

    def spaese_layer(self):
        prog = base.Program()
        scope = base.core.Scope()

        with base.scope_guard(scope):
            with base.program_guard(prog):
                input = paddle.static.data(
                    name="dnn_data", shape=[-1, 1], dtype="int64"
                )
                prob = ProbabilityEntry(0.5)
                emb = paddle.static.nn.sparse_embedding(
                    input=input,
                    size=[100, 10],
                    is_test=False,
                    entry=prob,
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
                entry = op.attr("entry")
                is_test = op.attr("is_test")
                is_sparse = op.attr("is_sparse")
                is_distributed = op.attr("is_distributed")

                self.assertEqual(entry, "probability_entry:0.5")
                self.assertTrue(is_distributed)
                self.assertTrue(is_sparse)
                self.assertFalse(is_test)


class TestEntryAttrs(EntryAttrChecks):
    def test_base(self):
        self.base()

    def test_prob(self):
        self.probability_entry()

    def test_counter(self):
        self.countfilter_entry()

    def test_showclick(self):
        self.showclick_entry()

    def test_spaese_embedding_layer(self):
        self.spaese_layer()


if __name__ == '__main__':
    unittest.main()

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
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import delete_ops

from transpiler_test import TranspilerTest


class TestDistTranspiler8192(TranspilerTest):
    def setUp(self):
        self.current_pserver_ep = "127.0.0.1:6174"
        self.min_block_size = 8192

    def test_transpiler(self):
        trainer = self.get_trainer(self.min_block_size)
        pserver, startup = self.get_pserver(self.current_pserver_ep,
                                            self.min_block_size)
        self.assertEqual([op.type for op in trainer.global_block().ops],
                         self.get_expect_trainer_ops())

        self.assertEqual(len(pserver.blocks), 3)
        # block0: listen_and_serv
        self.assertEqual([op.type for op in pserver.blocks[0].ops],
                         ["listen_and_serv"])
        # block2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[1].ops],
                         ["sum", "scale", "sgd"])

        # confirm startup program

        print "8192 start up program:", startup
        self.assertEqual([op.type for op in startup.global_block().ops], [
            "fill_constant", "fill_constant", "uniform_random", "uniform_random"
        ])

        # the variable #fc_w will be split into two blocks
        fc_w_var = startup.global_block().var("fc_w.block1")
        self.assertEqual(fc_w_var.shape, (500, 1000))

    def get_expect_trainer_ops(self):
        trainer = fluid.Program()

        with fluid.program_guard(trainer):
            optimize_ops, params_grads = self.net_conf()

        delete_ops(trainer.global_block(), optimize_ops)
        ops = [op.type for op in trainer.global_block().ops] + [
            "split_byref", "send", "send_barrier", "recv", "recv",
            "fetch_barrier", "concat"
        ]
        ops.insert(ops.index("elementwise_add_grad") + 1, "send")
        return ops


class TestDistTranspiler1048576(TranspilerTest):
    def setUp(self):
        self.current_pserver_ep = "127.0.0.1:6174"
        self.min_block_size = 1048576

    def test_transpiler_with_blocksize(self):
        trainer = self.get_trainer(self.min_block_size)

        pserver, startup = self.get_pserver(self.current_pserver_ep,
                                            self.min_block_size)
        pserver2, startup2 = self.get_pserver("127.0.0.1:6175",
                                              self.min_block_size)

        self.assertEqual(len(pserver.blocks), 2)

        print "tainer_ops:", self.get_expect_trainer_ops()
        self.assertEqual([op.type for op in trainer.global_block().ops],
                         self.get_expect_trainer_ops())

        # block0: listen_and_serv
        self.assertEqual([op.type for op in pserver.blocks[0].ops],
                         ["listen_and_serv"])

        # block2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[1].ops],
                         ["sum", "scale", "sgd"])

        # print "ops:", [op.type for op in startup.global_block().ops]
        # confirm startup program
        self.assertEqual([op.type for op in startup.global_block().ops],
                         ["fill_constant", "fill_constant"])

        # the variable #fc_w will not split into two blocks
        fc_w_var = startup2.global_block().var("fc_w")
        self.assertEqual(fc_w_var.shape, (1000L, 1000L))

    def get_expect_trainer_ops(self):
        trainer = fluid.Program()

        with fluid.program_guard(trainer):
            optimize_ops, params_grads = self.net_conf()

        delete_ops(trainer.global_block(), optimize_ops)

        ops = [op.type for op in trainer.global_block().ops] + [
            "send", "send_barrier", "recv", "recv", "fetch_barrier"
        ]

        ops.insert(ops.index("elementwise_add_grad") + 1, "send")
        return ops


if __name__ == "__main__":
    unittest.main()

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
from dist_transpiler_test_base import TranspilerTest


class TestBasicModelAsync(TranspilerTest):
    def transpiler_test_impl(self):
        config = fluid.DistributeTranspilerConfig()
        config.sync_mode = False
        config.runtime_split_send_recv = True

        pserver, startup = self.get_pserver(self.pserver1_ep, config, False)
        pserver2, startup2 = self.get_pserver(self.pserver2_ep, config, False)

        trainer, _ = self.get_trainer(config, False)
        self.assertEqual([op.type for op in trainer.global_block().ops], [
            'mul', 'elementwise_add', 'elementwise_sub', 'square', 'mean',
            'fill_constant', 'mean_grad', 'square_grad', 'elementwise_sub_grad',
            'elementwise_add_grad', 'send', 'mul_grad', 'send', 'recv', 'recv'
        ])
        self.assertEqual(len(pserver.blocks), 3)
        # block0: listen_and_serv
        self.assertEqual([op.type for op in pserver.blocks[0].ops],
                         ["listen_and_serv"])
        self.assertEqual(pserver.blocks[0].ops[0].attr("training_mode"), 2)
        # block1~2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[2].ops], ["sgd"])


class TestBasicModelHalfAsync(TranspilerTest):
    def transpiler_test_impl(self):
        config = fluid.DistributeTranspilerConfig()
        config.sync_mode = False
        config.runtime_split_send_recv = False

        pserver, startup = self.get_pserver(self.pserver1_ep, config, False)
        pserver2, startup2 = self.get_pserver(self.pserver2_ep, config, False)

        trainer, _ = self.get_trainer(config, False)
        self.assertEqual([op.type for op in trainer.global_block().ops], [
            'mul', 'elementwise_add', 'elementwise_sub', 'square', 'mean',
            'fill_constant', 'mean_grad', 'square_grad', 'elementwise_sub_grad',
            'elementwise_add_grad', 'send', 'mul_grad', 'split_byref', 'send',
            'recv', 'recv', 'concat'
        ])
        self.assertEqual(len(pserver.blocks), 3)
        # block0: listen_and_serv
        self.assertEqual([op.type for op in pserver.blocks[0].ops],
                         ["listen_and_serv"])
        self.assertEqual(pserver.blocks[0].ops[0].attr("training_mode"), 1)
        # block1~2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[2].ops], ["sgd"])


class TestBasicModelSync(TranspilerTest):
    def transpiler_test_impl(self):
        config = fluid.DistributeTranspilerConfig()
        config.sync_mode = True
        config.runtime_split_send_recv = False

        pserver, startup = self.get_pserver(self.pserver1_ep, config, True)
        pserver2, startup2 = self.get_pserver(self.pserver2_ep, config, True)

        trainer, _ = self.get_trainer(config, True)
        self.assertEqual([op.type for op in trainer.global_block().ops], [
            'mul', 'elementwise_add', 'elementwise_sub', 'square', 'mean',
            'fill_constant', 'mean_grad', 'square_grad', 'elementwise_sub_grad',
            'elementwise_add_grad', 'send', 'mul_grad', 'split_byref', 'send',
            'send_barrier', 'recv', 'recv', 'fetch_barrier', 'concat'
        ])

        self.assertEqual(len(pserver.blocks), 3)
        # block0: listen_and_serv
        self.assertEqual([op.type for op in pserver.blocks[0].ops],
                         ["listen_and_serv"])
        self.assertEqual(pserver.blocks[0].ops[0].attr("training_mode"), 0)
        # block1~2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[2].ops],
                         ["sum", "scale", "sgd"])


if __name__ == "__main__":
    unittest.main()

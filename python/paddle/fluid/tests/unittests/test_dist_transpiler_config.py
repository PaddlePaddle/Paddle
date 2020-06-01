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
import gc

gc.set_debug(gc.DEBUG_COLLECTABLE)


class TranspilerTest(unittest.TestCase):
    def setUp(self):
        self.trainer_id = 0
        self.trainers = 2
        self.pservers = 2
        # NOTE: we do not actually bind this port
        self.pserver_eps = "127.0.0.1:6174,127.0.0.1:6175"
        self.pserver1_ep = "127.0.0.1:6174"
        self.pserver2_ep = "127.0.0.1:6175"
        self.sync_mode = True
        self.transpiler = None

    def net_conf(self):
        x = fluid.layers.data(name='x', shape=[1000], dtype='float32')
        y_predict = fluid.layers.fc(input=x,
                                    size=1000,
                                    act=None,
                                    param_attr=fluid.ParamAttr(name='fc_w'),
                                    bias_attr=fluid.ParamAttr(name='fc_b'))
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        sgd_optimizer.minimize(avg_cost)

    def get_main_program(self):
        main = fluid.Program()
        main.random_seed = 1
        with fluid.program_guard(main):
            self.net_conf()
        self.origin_prog = main.clone()
        return main

    def get_trainer(self, config=None, sync_mode=True):
        src = fluid.default_startup_program().clone()

        t = self._transpiler_instance(config, sync_mode=True)

        trainer_main = t.get_trainer_program(wait_port=False)
        trainer_startup = fluid.default_startup_program()

        assert (src.num_blocks == 1)
        assert (trainer_startup.num_blocks == src.num_blocks)

        return trainer_main, trainer_startup

    def get_pserver(self, ep, config=None, sync_mode=True):
        t = self._transpiler_instance(config, sync_mode)
        pserver = t.get_pserver_program(ep)
        startup = t.get_startup_program(ep, pserver)
        return pserver, startup

    def _transpiler_instance(self, config=None, sync_mode=True):
        if not self.transpiler:
            main = self.get_main_program()
            self.transpiler = fluid.DistributeTranspiler(config=config)
            self.transpiler.transpile(
                self.trainer_id,
                program=main,
                pservers=self.pserver_eps,
                trainers=self.trainers,
                sync_mode=sync_mode)

        return self.transpiler

    def transpiler_test_impl(self):
        pass

    def test_transpiler(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                self.transpiler_test_impl()
        # NOTE: run gc.collect to eliminate pybind side objects to
        # prevent random double-deallocate when inherited in python.
        del self.transpiler
        del main
        del startup
        gc.collect()


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
        self.assertEqual(pserver.blocks[0].ops[0].attr("distributed_mode"), 1)
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
        self.assertEqual(pserver.blocks[0].ops[0].attr("distributed_mode"), 2)
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
        self.assertEqual(pserver.blocks[0].ops[0].attr("distributed_mode"), 0)
        # block1~2: optimize pass
        self.assertEqual([op.type for op in pserver.blocks[2].ops],
                         ["sum", "scale", "sgd"])


if __name__ == "__main__":
    unittest.main()

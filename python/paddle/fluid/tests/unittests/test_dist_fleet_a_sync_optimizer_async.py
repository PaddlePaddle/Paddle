# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import paddle
import paddle.distributed.fleet.base.role_maker as role_maker

paddle.enable_static()


class TestFleetGradientMergeMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"
        ] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_a_sync_optimizer_trainer(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        import paddle.distributed.fleet as fleet

        main_program = paddle.fluid.Program()
        startup_program = paddle.fluid.Program()

        paddle.fluid.framework.switch_main_program(main_program)
        paddle.fluid.framework.switch_startup_program(startup_program)

        fleet.init(role_maker.PaddleCloudRoleMaker())

        x = paddle.fluid.layers.data(name='x', shape=[1], dtype='float32')
        y = paddle.fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        strategy.a_sync_configs = {"launch_barrier": False}

        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        prog = paddle.fluid.default_main_program()
        self.assertEqual(prog.global_block().ops[-1].type, "send_barrier")

        sends = 0
        sgds = 0
        for op in prog.global_block().ops:
            if op.type == "send":
                sends += 1
            if op.type == "sgd":
                sgds += 1
        self.assertEqual(sends, 0)
        self.assertEqual(sgds, 0)

    def test_a_sync_optimizer_pserver(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        import paddle.distributed.fleet as fleet

        main_program = paddle.fluid.Program()
        startup_program = paddle.fluid.Program()

        paddle.fluid.framework.switch_main_program(main_program)
        paddle.fluid.framework.switch_startup_program(startup_program)

        fleet.init(role_maker.PaddleCloudRoleMaker())

        x = paddle.fluid.layers.data(name='x', shape=[1], dtype='float32')
        y = paddle.fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"launch_barrier": False}
        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        fleet.init_server()


if __name__ == "__main__":
    unittest.main()

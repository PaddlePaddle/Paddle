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

<<<<<<< HEAD
import os
import unittest

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid
import paddle.static as static
=======
import unittest
import paddle
import paddle.fluid as fluid
import paddle.static as static
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import os
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestFleetMetaOptimizer(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"
        ] = "127.0.0.1:36001,127.0.0.1:36002"

    def net(self):
        with static.device_guard("gpu:0"):
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32'
            )
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')
            input_z = paddle.static.data(
                name="z", shape=[-1, 1], dtype="float32"
            )
            with static.device_guard("gpu:all"):
                input_z = input_z * 1.0
                input_z.stop_gradient = True
            fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
            fc_1 = fc_1 * input_z

        with static.device_guard("gpu:1"):
            fc_2 = paddle.static.nn.fc(x=fc_1, size=64, activation='tanh')
            # for pipeline check_pipeline_persist_var coverage
            fc_2.persistable = True
            fc_2 = fc_2 * input_z
            prediction = paddle.static.nn.fc(
                x=[fc_2], size=2, activation='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
=======

    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def net(self):
        with static.device_guard("gpu:0"):
            input_x = paddle.fluid.layers.data(name="x",
                                               shape=[32],
                                               dtype='float32')
            input_y = paddle.fluid.layers.data(name="y",
                                               shape=[1],
                                               dtype='int64')
            input_z = paddle.fluid.layers.data(name="z",
                                               shape=[1],
                                               dtype="float32")
            with static.device_guard("gpu:all"):
                input_z = input_z * 1.0
                input_z.stop_gradient = True
            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_1 = fc_1 * input_z

        with static.device_guard("gpu:1"):
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            # for pipeline check_pipeline_persist_var coverage
            fc_2.persistable = True
            fc_2 = fc_2 * input_z
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(input=prediction,
                                                     label=input_y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_cost = paddle.mean(x=cost)
        return avg_cost

    def test_pipeline_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.pipeline = True
        strategy.pipeline_configs = {
            'micro_batch_size': 1,
<<<<<<< HEAD
            'accumulate_steps': 2,
=======
            'accumulate_steps': 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        train_prog, startup_prog = static.Program(), static.Program()
        with static.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                avg_cost = self.net()

                optimizer = paddle.fluid.optimizer.Adam(0.01)
<<<<<<< HEAD
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=strategy
                )
                optimizer.minimize(avg_cost)

    def test_pipeline_amp_optimizer(self):
        """test pipeline&amp with device:all"""
=======
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        strategy=strategy)
                optimizer.minimize(avg_cost)

    def test_pipeline_amp_optimizer(self):
        """ test pipeline&amp with device:all """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.amp = True
        strategy.pipeline = True
        strategy.pipeline_configs = {
            'micro_batch_size': 1,
<<<<<<< HEAD
            'accumulate_steps': 2,
=======
            'accumulate_steps': 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        train_prog, startup_prog = static.Program(), static.Program()
        with static.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                avg_cost = self.net()

                optimizer = paddle.fluid.optimizer.Adam(0.01)
<<<<<<< HEAD
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=strategy
                )
=======
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        strategy=strategy)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                optimizer.minimize(avg_cost)

        ops = train_prog._pipeline_opt['section_program'].global_block().ops
        ops = [op.type for op in ops]
        self.assertEqual(ops.count('send_v2'), 1)
        self.assertEqual(ops.count('recv_v2'), 1)


if __name__ == "__main__":
    unittest.main()

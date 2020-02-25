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

import unittest
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet, TrainStatus
import os


class FleetTest(unittest.TestCase):
    def test_check_point(self):
        dir_path = "./my_paddle_model"
        file_name = "persistables"

        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        feeder = fluid.DataFeeder(
            feed_list=[image, label], place=fluid.CPUPlace())
        predict = fluid.layers.fc(input=image, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        avg_loss = fluid.layers.mean(loss)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)

        dist_optimizer = fleet.distributed_optimizer(optimizer)
        dist_optimizer.minimize(avg_loss)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        status = TrainStatus()
        status.epoch_no = 2
        fleet.save_check_point(exe, dir_path, train_status=status)

        status2 = fleet.load_check_point(exe, dir_path)
        assert status2 == status, "Checkpoint error!"


"""
class FleetCollectiveTest(unittest.TestCase):
    def test_open_sync_batch_norm(self):
        import paddle.fluid as fluid
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

        if not fluid.core.is_compiled_with_cuda():
            # Operator "gen_nccl_id" has not been registered
            return

        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)

        optimizer = fluid.optimizer.AdamOptimizer()

        role = role_maker.UserDefinedCollectiveRoleMaker(0, ['127.0.0.1:6170'])
        fleet.init(role)

        dist_strategy = DistributedStrategy()
        #dist_strategy.sync_batch_norm = True

        dist_optimizer = fleet.distributed_optimizer(
            optimizer, strategy=dist_strategy)
        dist_optimizer.minimize(loss)

        #self.assertEqual(dist_strategy.exec_strategy.num_threads, 1)
"""

if __name__ == '__main__':
    unittest.main()

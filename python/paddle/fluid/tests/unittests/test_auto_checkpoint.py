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
import sys

from paddle.fluid.incubate.fleet.utils.fs import LocalFS
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient


class FleetTest(unittest.TestCase):
    def _init_model(self):
        # Set place explicitly.
        use_cuda = True
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        optimizer = None

        train_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(train_program, startup_program):
            data = fluid.data(name='X', shape=[None, 1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            optimizer = fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

            # Run the startup program once and only once.
            # Not need to optimize/compile the startup program.
            startup_program.random_seed = 1
            exe.run(startup_program)

        return exe, optimizer, loss, train_program

    def _run(self, exe, main_program):
        for i in fluid.train_eoch_range(10):
            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss.name])
                print("fetch:", loss)

            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss.name])
                print("fetch:", loss)

        assert acp._get_train_epoch_range() == None

        a = []
        for i in fluid.train_eoch_range(10):
            a.append(i)
            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss.name])
                print("fetch:", loss)
        assert len(a) == 10, "a must run from 0 to 9"

    def test_without_fleet(self):
        exe, _, loss, main_program = self._init_model()
        self._run(exe, main_program)

    def test_with_fleet(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        exe, optimizer, avg_loss, main_program = self._init_model()

        dist_optimizer = fleet.distributed_optimizer(optimizer)
        dist_optimizer.minimize(avg_loss)

        self._run(exe, fleet.main_program)


if __name__ == '__main__':
    unittest.main()

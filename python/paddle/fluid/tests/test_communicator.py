#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import time
import threading
import numpy

import paddle
import paddle.fluid as fluid
from paddle.fluid.communicator import Communicator
from paddle.fluid.communicator import AsyncMode

import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


class TestCommunicator(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost

    def test_communicator_init_and_start(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])

        fleet.init(role)
        avg_cost = self.net()

        optimizer = fluid.optimizer.SGD(0.01)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = True
        strategy.wait_port = False
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        comm = Communicator(fleet.main_program, AsyncMode.ASYNC)
        comm.start()
        time.sleep(10)
        comm.stop()


class TestCommunicator2(unittest.TestCase):
    def test_communicator_init_and_start(self):
        prog = fluid.Program()
        comm = Communicator(prog, AsyncMode.ASYNC)
        comm.start()
        comm.stop()


class TestCommunicatorHalfAsync(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost, x, y

    def test_communicator_init_and_start(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])

        fleet.init(role)
        avg_cost, x, y = self.net()

        optimizer = fluid.optimizer.SGD(0.01)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False 
        strategy.runtime_split_send_recv = True
        strategy.half_async = True
        strategy.wait_port = False
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        def reader():
            for i in range(1000):
                x = numpy.random.random((1, 13)).astype('float32')
                y = numpy.random.randint(0, 2, (1, 1)).astype('int64')
                return x,y

        def run_trainer(startup, main):
            place = fluid.core.CPUPlace()
            exe = fluid.Executor(place)

            op_ids = []
            for op in startup.global_block().ops:
                if op.type == "recv" or op.type == "fetch_barrier":
                    op_ids.append(startup.global_block().ops.index(op))

            for idx in op_ids[::-1]:
                startup.global_block()._remove_op(idx)

            with open("xx.start", "w") as wb:
                wb.write(str(startup))

            exe.run(startup)

            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            data = reader()

            exe.run(main, feed=feeder.feed([data]), fetch_list=[])

        comm = Communicator(fleet.main_program, AsyncMode.HALF_ASYNC)
        comm.start()

        t = threading.Thread(target=run_trainer, args=(fleet.startup_program, fleet.main_program))
        t.start()
        #t.join()

        time.sleep(10)
        comm.stop()


class TestCommunicatorHalfAsync2(unittest.TestCase):
    def test_communicator_init_and_start(self):
        prog = fluid.Program()
        comm = Communicator(prog, AsyncMode.HALF_ASYNC)
        comm.start()
        comm.stop()


if __name__ == '__main__':
    unittest.main()

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

import os
import unittest
import time
import threading
import numpy

import paddle
paddle.enable_static()

import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet


class TestCommunicator(unittest.TestCase):
    def test_communicator_ps_gpu(self):
        with open("test_communicator_ps_gpu.txt", "w") as f:
            data = "1 0.6 1 0.7\n"
            f.write(data)

        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002,127.0.0.2:36002"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["FLAGS_selected_gpus"] = "0"
        role = role_maker.PaddleCloudRoleMaker()

        fleet.init(role)
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        slots_vars = [x, y]

        cost = fluid.layers.square_error_cost(input=x, label=y)
        avg_cost = fluid.layers.mean(cost)

        optimizer = fluid.optimizer.Adam(0.01)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {
            "launch_barrier": False,
            "use_ps_gpu": 1,
        }
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        dataset = paddle.distributed.InMemoryDataset()
        dataset.init(
            batch_size=32, thread_num=1, pipe_command="cat", use_var=slots_vars)
        dataset.set_filelist(["test_communicator_ps_gpu.txt"])
        dataset.set_date("20211111")
        dataset.load_into_memory(is_shuffle=True)

        os.environ["TEST_MODE"] = "1"
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        main_program._fleet_opt = {"stat_var_names": [x.name]}
        fleet.init_worker()

        try:
            exe.train_from_dataset(main_program, dataset)
        except ImportError as e:
            pass
        except Exception as e:
            self.assertTrue(False)
        time.sleep(10)
        fleet.stop_worker()
        os.remove("./test_communicator_ps_gpu.txt")


if __name__ == '__main__':
    unittest.main()

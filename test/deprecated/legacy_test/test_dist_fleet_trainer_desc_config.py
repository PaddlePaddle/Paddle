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

os.environ["WITH_DISTRIBUTE"] = "ON"
os.environ['FLAGS_enable_pir_api'] = '0'

import paddle
from paddle.distributed.fleet.base import role_maker

paddle.enable_static()


class TestDistStrategyTrainerDescConfig(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = (
            "127.0.0.1:36001,127.0.0.2:36001"
        )

    def test_trainer_desc_config(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        from paddle.distributed import fleet

        fleet.init(role_maker.PaddleCloudRoleMaker())

        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"launch_barrier": 0}
        config = {
            "dump_fields_path": "dump_data",
            "dump_fields": ["xxx", "yyy"],
            "dump_param": ['zzz'],
        }
        strategy.trainer_desc_configs = config

        optimizer = paddle.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        program = paddle.static.default_main_program()
        self.assertEqual(program._fleet_opt["dump_fields_path"], "dump_data")
        self.assertEqual(len(program._fleet_opt["dump_fields"]), 2)
        self.assertEqual(len(program._fleet_opt["dump_param"]), 1)
        self.assertEqual(
            program._fleet_opt["mpi_size"],
            int(os.environ["PADDLE_TRAINERS_NUM"]),
        )

        optimizer = paddle.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize([avg_cost])

        program = avg_cost.block.program
        self.assertEqual(program._fleet_opt["dump_fields_path"], "dump_data")
        self.assertEqual(len(program._fleet_opt["dump_fields"]), 2)
        self.assertEqual(len(program._fleet_opt["dump_param"]), 1)
        self.assertEqual(
            program._fleet_opt["mpi_size"],
            int(os.environ["PADDLE_TRAINERS_NUM"]),
        )


if __name__ == "__main__":
    unittest.main()

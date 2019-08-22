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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import os
import signal
import subprocess
import time
import unittest
from op_test import OpTest
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.trainer_desc import DistMultiTrainer
from paddle.fluid.device_worker import DownpourSGD


class TestListenAndServOp(OpTest):
    def setUp(self):
        self.ps_timeout = 5
        self.ip = "127.0.0.1"

    def test_device_work_use_cvm(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        x_emb = fluid.layers.embedding(
            input=x, size=[1, 2], is_distributed=True)
        y_predict = fluid.layers.fc(input=x_emb, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        fleet.init()
        adam = fluid.optimizer.Adam(learning_rate=0.000005)
        adam = fleet.distributed_optimizer(adam, strategy={"use_cvm": True})
        adam.minimize([avg_cost])
        opt_info = fleet._opt_info
        opt_info[
            "fleet_desc"].fs_client_param.uri = "afs://tianqi.afs.baidu.com:9902"
        opt_info["fleet_desc"].fs_client_param.user = "fcr-tianqi-d"
        opt_info["fleet_desc"].fs_client_param.passwd = "absUPEwUB7nc"
        opt_info[
            "fleet_desc"].fs_client_param.hadoop_bin = "$HADOOP_HOME/bin/hadoop"
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        trainer = DistMultiTrainer()
        trainer._set_program(fluid.default_main_program())
        device_worker = DownpourSGD()
        device_worker._set_fleet_desc(opt_info["fleet_desc"])
        trainer._set_device_worker(device_worker)
        trainer._set_fleet_desc(opt_info["fleet_desc"])
        trainer._gen_trainer_desc()

    def test_device_work(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        x_emb = fluid.layers.embedding(
            input=x, size=[1, 2], is_distributed=True)
        y_predict = fluid.layers.fc(input=x_emb, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        fleet.init()
        adam = fluid.optimizer.Adam(learning_rate=0.000005)
        adam = fleet.distributed_optimizer(adam, strategy=dict())
        adam.minimize([avg_cost])
        opt_info = fleet._opt_info
        opt_info[
            "fleet_desc"].fs_client_param.uri = "afs://tianqi.afs.baidu.com:9902"
        opt_info["fleet_desc"].fs_client_param.user = "fcr-tianqi-d"
        opt_info["fleet_desc"].fs_client_param.passwd = "absUPEwUB7nc"
        opt_info[
            "fleet_desc"].fs_client_param.hadoop_bin = "$HADOOP_HOME/bin/hadoop"
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        trainer = DistMultiTrainer()
        trainer._set_program(fluid.default_main_program())
        device_worker = DownpourSGD()
        device_worker._set_fleet_desc(opt_info["fleet_desc"])
        trainer._set_device_worker(device_worker)
        trainer._set_fleet_desc(opt_info["fleet_desc"])
        trainer._gen_trainer_desc()


if __name__ == "__main__":
    unittest.main()

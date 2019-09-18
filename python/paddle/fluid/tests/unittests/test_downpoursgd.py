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
import sys
from op_test import OpTest
from paddle.fluid.trainer_desc import DistMultiTrainer
from paddle.fluid.device_worker import DownpourSGD
from google.protobuf import text_format
import paddle.fluid.incubate.fleet.parameter_server.pslib.ps_pb2 as pslib


class TestListenAndServOp(OpTest):
    def setUp(self):
        pass

    def test_device_work_use_cvm(self):
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            cmd = "wget --no-check-certificate https://pslib.bj.bcebos.com/fleet_desc.prototxt"
            os.system(cmd)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x_emb = fluid.layers.embedding(
                input=x, size=[1, 2], is_distributed=True)
            y_predict = fluid.layers.fc(input=x_emb, size=1, act=None)
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

            ps_param = pslib.PSParameter()
            with open("fleet_desc.prototxt") as f:
                text_format.Merge(f.read(), ps_param)
            fleet_desc = ps_param
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            opt_info = {}
            main_program = fluid.default_main_program()
            program_id = str(id(avg_cost.block.program))
            program_configs = {}
            program_configs[program_id] = {
                "pull_sparse": [0],
                "push_sparse": [0]
            }
            program_configs[program_id]["pull_dense"] = [1]
            program_configs[program_id]["push_dense"] = [1]

            worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
            opt_info["program_configs"] = program_configs
            opt_info["trainer"] = "DistMultiTrainer"
            opt_info["device_worker"] = "DownpourSGD"
            opt_info["optimizer"] = "DownpourSGD"
            opt_info["fleet_desc"] = ps_param
            opt_info["worker_skipped_ops"] = worker_skipped_ops
            opt_info["use_cvm"] = True
            opt_info["scale_datanorm"] = -1
            opt_info["dump_slot"] = False
            opt_info["stat_var_names"] = []

            main_program._fleet_opt = opt_info
            trainer = DistMultiTrainer()
            trainer._set_program(main_program)
            device_worker = DownpourSGD()
            device_worker._set_fleet_desc(fleet_desc)
            trainer._set_device_worker(device_worker)
            trainer._set_fleet_desc(fleet_desc)
            trainer._gen_trainer_desc()
            cmd = "rm fleet_desc.prototxt*"
            os.system(cmd)

    def test_device_work(self):
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            cmd = "wget --no-check-certificate https://pslib.bj.bcebos.com/fleet_desc.prototxt"
            os.system(cmd)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x_emb = fluid.layers.embedding(
                input=x, size=[1, 2], is_distributed=True)
            y_predict = fluid.layers.fc(input=x_emb, size=1, act=None)
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

            ps_param = pslib.PSParameter()
            with open("fleet_desc.prototxt") as f:
                text_format.Merge(f.read(), ps_param)
            fleet_desc = ps_param
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            opt_info = {}
            main_program = fluid.default_main_program()
            program_id = str(id(avg_cost.block.program))
            program_configs = {}
            program_configs[program_id] = {
                "pull_sparse": [0],
                "push_sparse": [0]
            }
            program_configs[program_id]["pull_dense"] = [1]
            program_configs[program_id]["push_dense"] = [1]

            worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
            opt_info["program_configs"] = program_configs
            opt_info["trainer"] = "DistMultiTrainer"
            opt_info["device_worker"] = "DownpourSGD"
            opt_info["optimizer"] = "DownpourSGD"
            opt_info["fleet_desc"] = ps_param
            opt_info["worker_skipped_ops"] = worker_skipped_ops
            opt_info["use_cvm"] = False
            opt_info["scale_datanorm"] = -1
            opt_info["dump_slot"] = False
            opt_info["stat_var_names"] = []

            main_program._fleet_opt = opt_info
            trainer = DistMultiTrainer()
            trainer._set_program(main_program)
            device_worker = DownpourSGD()
            device_worker._set_fleet_desc(fleet_desc)
            trainer._set_device_worker(device_worker)
            trainer._set_fleet_desc(fleet_desc)
            trainer._gen_trainer_desc()
            cmd = "rm fleet_desc.prototxt*"
            os.system(cmd)


if __name__ == "__main__":
    unittest.main()

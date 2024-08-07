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
"""Test cases for Downpour."""

import os
import sys
import unittest

from google.protobuf import text_format

import paddle
import paddle.incubate.distributed.fleet.parameter_server.pslib.ps_pb2 as pslib
from paddle import base
from paddle.base.trainer_factory import TrainerFactory
from paddle.incubate.distributed.fleet.parameter_server.pslib.node import (
    DownpourServer,
    DownpourWorker,
)

cache_path = os.path.expanduser('~/.cache/paddle/dataset')


class TestListenAndServOp(unittest.TestCase):
    """This class is Test Listen And ServOp."""

    def setUp(self):
        """This function is set Up."""
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

    def test_device_work_use_cvm(self):
        """test device work use_cvm."""
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            if not os.path.exists(
                '{}/{}'.format(cache_path, 'fleet_desc.prototxt')
            ):
                cmd = f"wget --no-check-certificate https://pslib.bj.bcebos.com/fleet_desc.prototxt -P {cache_path}/"
                os.system(cmd)
            x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64')
            x_emb = paddle.static.nn.embedding(
                input=x, size=[1, 2], is_distributed=True
            )
            y_predict = paddle.static.nn.fc(x=x_emb, size=1)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            ps_param = pslib.PSParameter()
            with open(f"{cache_path}/fleet_desc.prototxt") as f:
                text_format.Merge(f.read(), ps_param)
            fleet_desc = ps_param
            exe = base.Executor(base.CPUPlace())
            exe.run(base.default_startup_program())

            opt_info = {}
            main_program = base.default_main_program()
            program_id = str(id(avg_cost.block.program))
            program_configs = {}
            program_configs[program_id] = {
                "pull_sparse": [0],
                "push_sparse": [0],
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
            worker = DownpourWorker(None)
            server = DownpourServer()
            server.add_sparse_table(0, {})
            worker.get_desc().CopyFrom(ps_param.trainer_param[0])
            opt_info["program_id_to_worker"] = {program_id: worker}

            main_program._fleet_opt = opt_info
            trainer = TrainerFactory()._create_trainer(main_program._fleet_opt)
            trainer._set_program(main_program)
            trainer._gen_trainer_desc()

    def test_device_work(self):
        """This function is test devicve worker."""
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            if not os.path.exists(
                '{}/{}'.format(cache_path, 'fleet_desc.prototxt')
            ):
                cmd = f"wget --no-check-certificate https://pslib.bj.bcebos.com/fleet_desc.prototxt -P {cache_path}/"
                os.system(cmd)
            x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64')
            x_emb = paddle.static.nn.embedding(
                input=x, size=[1, 2], is_distributed=True
            )
            y_predict = paddle.static.nn.fc(x=x_emb, size=1)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            ps_param = pslib.PSParameter()
            with open(f"{cache_path}/fleet_desc.prototxt") as f:
                text_format.Merge(f.read(), ps_param)
            fleet_desc = ps_param
            exe = base.Executor(base.CPUPlace())
            exe.run(base.default_startup_program())

            opt_info = {}
            main_program = base.default_main_program()
            program_id = str(id(avg_cost.block.program))
            program_configs = {}
            program_configs[program_id] = {
                "pull_sparse": [0],
                "push_sparse": [0],
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
            worker = DownpourWorker(None)
            worker.get_desc().CopyFrom(ps_param.trainer_param[0])
            opt_info["program_id_to_worker"] = {program_id: worker}

            main_program._fleet_opt = opt_info
            trainer = TrainerFactory()._create_trainer(main_program._fleet_opt)
            trainer._set_program(main_program)
            trainer._gen_trainer_desc()

    def test_downpour_opt_work(self):
        """This function is test devicve worker."""
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            if not os.path.exists(
                '{}/{}'.format(cache_path, 'fleet_desc.prototxt')
            ):
                cmd = f"wget --no-check-certificate https://pslib.bj.bcebos.com/fleet_desc.prototxt -P {cache_path}/"
                os.system(cmd)
            x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64')
            x_emb = paddle.static.nn.embedding(
                input=x, size=[1, 2], is_distributed=True
            )
            y_predict = paddle.static.nn.fc(x=x_emb, size=1)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            ps_param = pslib.PSParameter()
            with open(f"{cache_path}/fleet_desc.prototxt") as f:
                text_format.Merge(f.read(), ps_param)
            fleet_desc = ps_param
            exe = base.Executor(base.CPUPlace())
            exe.run(base.default_startup_program())

            opt_info = {}
            main_program = base.default_main_program()
            program_id = str(id(avg_cost.block.program))
            program_configs = {}
            program_configs[program_id] = {
                "pull_sparse": [0],
                "push_sparse": [0],
            }
            program_configs[program_id]["pull_dense"] = [1]
            program_configs[program_id]["push_dense"] = [1]

            worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
            opt_info["program_configs"] = program_configs
            opt_info["trainer"] = "DistMultiTrainer"
            opt_info["device_worker"] = "DownpourSGDOPT"
            opt_info["optimizer"] = "DownpourSGD"
            opt_info["fleet_desc"] = ps_param
            opt_info["worker_skipped_ops"] = worker_skipped_ops
            opt_info["use_cvm"] = False
            opt_info["scale_datanorm"] = -1
            opt_info["dump_slot"] = False
            opt_info["stat_var_names"] = []
            opt_info["user_define_dump_filename"] = "./dump_filename/dump.txt"
            worker = DownpourWorker(None)
            worker.get_desc().CopyFrom(ps_param.trainer_param[0])
            opt_info["program_id_to_worker"] = {program_id: worker}

            main_program._fleet_opt = opt_info
            trainer = TrainerFactory()._create_trainer(main_program._fleet_opt)
            trainer._set_program(main_program)
            trainer._gen_trainer_desc()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

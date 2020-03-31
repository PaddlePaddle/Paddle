# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig
from paddle.fluid.transpiler.geo_sgd_transpiler import GeoSgdTranspiler
from test_dist_fleet_base import TestFleetBase
from dist_simnet_bow import train_network


class TestDistGeoCtr_2x2(TestFleetBase):
    def _setup_config(self):
        self._mode = "geo"
        self._reader = "dataset"
        self._geo_sgd_need_push_nums = 5

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": ""
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "4"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_ctr.py", delta=1e-5, check_error_log=True)


class TestGeoSgdTranspiler(unittest.TestCase):
    def test_pserver(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])

        fleet.init(role)

        batch_size = 128
        is_sparse = True
        is_distribute = False

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        strategy.geo_sgd_mode = True
        strategy.geo_sgd_need_push_nums = 5

        avg_cost, _, _ = train_network(batch_size, is_distribute, is_sparse)

        optimizer = fluid.optimizer.SGD(0.1)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        pserver_startup_program = fleet.startup_program
        pserver_mian_program = fleet.main_program


if __name__ == "__main__":
    unittest.main()

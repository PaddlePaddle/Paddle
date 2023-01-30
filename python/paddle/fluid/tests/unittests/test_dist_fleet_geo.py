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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import os

os.environ["WITH_DISTRIBUTE"] = "ON"
import unittest
<<<<<<< HEAD

from dist_fleet_simnet_bow import train_network
from test_dist_fleet_base import TestFleetBase

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid
=======
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker

from test_dist_fleet_base import TestFleetBase
from dist_fleet_simnet_bow import train_network
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestDistGeoCtr_2x2(TestFleetBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._mode = "geo"
        self._reader = "pyreader"
        self._geo_sgd_need_push_nums = 5

<<<<<<< HEAD
    def check_with_place(
        self, model_file, delta=1e-3, check_error_log=False, need_envs={}
    ):
=======
    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "LOG_DIRNAME": "/tmp",
            "LOG_PREFIX": self.__class__.__name__,
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "4"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
<<<<<<< HEAD
        self.check_with_place(
            "dist_fleet_ctr.py", delta=1e-5, check_error_log=False
        )


class TestGeoSgdTranspiler(unittest.TestCase):
=======
        self.check_with_place("dist_fleet_ctr.py",
                              delta=1e-5,
                              check_error_log=False)


class TestGeoSgdTranspiler(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_pserver(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
<<<<<<< HEAD
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"],
        )
=======
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        fleet.init(role)

        batch_size = 128
        is_sparse = True
        is_distribute = False

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": 100, "launch_barrier": False}

        avg_cost, _, _, _ = train_network(batch_size, is_distribute, is_sparse)

        optimizer = fluid.optimizer.SGD(0.1)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)


if __name__ == "__main__":
    unittest.main()

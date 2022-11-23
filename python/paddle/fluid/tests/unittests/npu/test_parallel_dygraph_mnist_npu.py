# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

sys.path.append("..")

from test_dist_base import TestDistBase
import paddle.fluid as fluid

flag_name = os.path.splitext(__file__)[0]
rank_table_file = b"""{
    "status": "completed",
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "127.0.0.1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.1.184.23",
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.2.21.93",
                    "rank_id": "1"
                }
            ]
        }
    ]
}"""

need_envs = {
    "ASCEND_AICPU_PATH":
    os.getenv("ASCEND_AICPU_PATH", "/usr/local/Ascend/nnae/latest"),
    "ASCEND_OPP_PATH":
    os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/nnae/latest/opp"),
    "HCCL_CONNECT_TIMEOUT":
    "7200",
    "HCCL_WHITELIST_DISABLE":
    "1",
    "HCCL_SECURITY_MODE":
    "1",
    "RANK_TABLE_FILE":
    "rank_table_file.json",
}


class TestParallelDygraphMnistNPU(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._hccl_mode = True
        self._dygraph = True
        self._enforce_place = "NPU"

    def test_mnist(self):
        with open("rank_table_file.json", "wb") as f:
            f.write(rank_table_file)
        if fluid.core.is_compiled_with_npu():
            self.check_with_place(
                os.path.abspath('../parallel_dygraph_mnist.py'),
                delta=1e-3,
                check_error_log=True,
                need_envs=need_envs,
                log_name=flag_name)


class TestFleetDygraphMnistNPU(TestParallelDygraphMnistNPU):

    def _setup_config(self):
        self._sync_mode = False
        self._hccl_mode = True
        self._dygraph = True
        self._enforce_place = "NPU"
        self._use_fleet_api = True


if __name__ == "__main__":
    unittest.main()

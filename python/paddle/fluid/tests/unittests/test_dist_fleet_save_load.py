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

import sys
import os
import shutil
import unittest
import tempfile
import json
import subprocess
import numpy as np

from test_dist_fleet_base import TestFleetBase, RUN_STEP


class TestFleetSaveLoadDense2x2(TestFleetBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"
        self._test_mode = 0  # 0: local save/load  1: hadoop save/load
        self.hadoop_path = "./fake_local_hadoop.py"

    def check_with_place(self, model_file, check_error_log=False, need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "10000",  # 5sec to fail fast
            "http_proxy": ""
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        required_envs.update(need_envs)

        if self._test_mode == 0:
            model_dir = tempfile.mkdtemp()
        else:
            model_dir = 'hdfs:/user/simnet_bow/'

        cluster_env = {}
        cluster_env["SAVE"] = "1"
        cluster_env["MODEL_DIR"] = model_dir
        cluster_env.update(required_envs)
        tr0_var, tr1_var = self._run_cluster(model_file, cluster_env)

        cluster_env = {}
        cluster_env["LOAD"] = "1"
        cluster_env["MODEL_DIR"] = model_dir
        cluster_env.update(required_envs)
        tr0_var, tr1_var = self._run_cluster(model_file, cluster_env)

        if model_dir.startswith('hdfs:'):
            model_dir = tempfile.gettempdir(
            ) + '/fake_hadoop_repos/user/simnet_bow/'
        cmd = 'ls ' + model_dir + ' | wc -l'
        cmd_exec = os.popen(cmd)
        self.assertEqual(cmd_exec.read().strip(), '6')
        cmd_exec.close()

    def test_fleet_local_save(self):
        need_envs = {
            "IS_DISTRIBUTED": '1',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self._test_mode = 0
        self.check_with_place(
            "fleet_save_load.py", check_error_log=False, need_envs=need_envs)

    def test_fleet_hadoop_save(self):
        need_envs = {
            "IS_DISTRIBUTED": '1',
            "IS_SPARSE": '1',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self._test_mode = 1
        self.check_with_place(
            "fleet_save_load.py", check_error_log=False, need_envs=need_envs)


if __name__ == "__main__":
    unittest.main()

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

import os
import shutil
import unittest
import tempfile

import numpy as np

from test_dist_base import TestDistBase, RUN_STEP


class TestDistSaveLoadDense2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):

        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "http_proxy": ""
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "7"
            required_envs["GLOG_logtostderr"] = "1"

        model_dir = tempfile.mkdtemp()

        local_env = {}
        local_env["SAVE"] = "1"
        local_env["MODEL_DIR"] = model_dir
        local_env.update(required_envs)

        cluster_env = {}
        cluster_env["LOAD"] = "1"
        cluster_env["MODEL_DIR"] = model_dir
        cluster_env.update(required_envs)

        local_var = self._run_local(model_file, local_env, check_error_log)
        tr0_var, tr1_var = self._run_cluster(model_file, cluster_env,
                                             check_error_log)

        shutil.rmtree(model_dir)

        local_np = np.array(eval(local_var[0]))
        train0_np = np.array(eval(tr0_var[0]))
        train1_np = np.array(eval(tr1_var[0]))
        self.assertAlmostEqual(local_np.all(), train0_np.all(), delta=delta)
        self.assertAlmostEqual(local_np.all(), train1_np.all(), delta=delta)
        self.assertAlmostEqual(train0_np.all(), train1_np.all(), delta=delta)

    @unittest.skip(reason="CI fail")
    def test_dist(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '0',
            'IS_SELF_CONTAINED_LR': '1'
        }
        self.check_with_place(
            "dist_save_load.py",
            delta=0,
            check_error_log=False,
            need_envs=need_envs)


if __name__ == "__main__":
    unittest.main()

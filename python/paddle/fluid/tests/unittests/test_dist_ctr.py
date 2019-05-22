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
import unittest
import tempfile
import shutil
from test_dist_base import TestDistBase

class TestDistCTR2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        self.check_with_place("dist_ctr.py", delta=1e-7, check_error_log=False)


class TestDistCTRWithL2Decay2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_dist_ctr(self):
        need_envs = {"USE_L2_DECAY": "1"}
        self.check_with_place(
            "dist_ctr.py",
            delta=1e-7,
            check_error_log=False,
            need_envs=need_envs)

class TestDistCTRSaveModelParams(TestDistBase):
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
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        model_dir = tempfile.mkdtemp()

        save_env = {}
        save_env["MODEL_DIR"] = model_dir
        save_env.update(required_envs)

        self._run_cluster(model_file, save_env, check_error_log)
        self.assertTrue(os.path.exists(model_dir + "/" + "__lookup_table__/wide_embedding_0"))
        self.assertTrue(os.path.exists(model_dir + "/" + "__lookup_table__/wide_embedding_1"))
        self.assertTrue(os.path.exists(model_dir + "/" + "__lookup_table__/deep_embedding_0"))
        self.assertTrue(os.path.exists(model_dir + "/" + "__lookup_table__/deep_embedding_1"))

        shutil.rmtree(model_dir)

    def test_dist_ctr_save(self):
        self.check_with_place(
            "dist_ctr_save.py",
            delta=1e-7,
            check_error_log=False)

if __name__ == "__main__":
    unittest.main()

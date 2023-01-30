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
<<<<<<< HEAD

import os
import shutil
import tempfile
import unittest

import numpy as np
from test_dist_base import TestDistBase
=======
from __future__ import print_function

import os
import shutil
import unittest
import tempfile

import numpy as np

from test_dist_base import TestDistBase, RUN_STEP

import os
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

flag_name = os.path.splitext(__file__)[0]


class TestDistSaveLoadDense2x2(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

<<<<<<< HEAD
    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
=======
    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={},
                         log_name=""):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
<<<<<<< HEAD
            "http_proxy": "",
=======
            "http_proxy": ""
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        required_envs.update(need_envs)

        if check_error_log:
<<<<<<< HEAD
            required_envs[
                "GLOG_vmodule"
            ] = "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10"
=======
            required_envs["GLOG_vmodule"] = \
                "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        tr0_var, tr1_var = self._run_cluster(
            model_file, cluster_env, check_error_log, log_name=flag_name
        )
=======
        tr0_var, tr1_var = self._run_cluster(model_file,
                                             cluster_env,
                                             check_error_log,
                                             log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        shutil.rmtree(model_dir)

        local_np = np.array(local_var)
        train0_np = np.array(tr0_var)
        train1_np = np.array(tr1_var)

        np.testing.assert_almost_equal(local_np, train0_np, decimal=2)
        np.testing.assert_almost_equal(local_np, train1_np, decimal=2)
        np.testing.assert_almost_equal(train0_np, train1_np, decimal=2)

    def test_dist(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '0',
            'IS_SELF_CONTAINED_LR': '1',
            'SAVE_MODE': 'LOCAL',
        }
<<<<<<< HEAD
        self.check_with_place(
            "dist_save_load.py",
            delta=0,
            check_error_log=False,
            need_envs=need_envs,
        )


class TestDistSaveLoadWithPServerStateDense2x2(TestDistBase):
=======
        self.check_with_place("dist_save_load.py",
                              delta=0,
                              check_error_log=False,
                              need_envs=need_envs)


class TestDistSaveLoadWithPServerStateDense2x2(TestDistBase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

<<<<<<< HEAD
    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
=======
    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={},
                         log_name=""):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
<<<<<<< HEAD
            "http_proxy": "",
=======
            "http_proxy": ""
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        required_envs.update(need_envs)

        if check_error_log:
<<<<<<< HEAD
            required_envs[
                "GLOG_vmodule"
            ] = "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10"
=======
            required_envs["GLOG_vmodule"] = \
                "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            required_envs["GLOG_logtostderr"] = "1"

        model_dir = tempfile.mkdtemp()

        save_env = {}
        save_env["SAVE_MODE"] = "DIST"
        save_env["SAVE"] = "1"
        save_env["MODEL_DIR"] = model_dir
        save_env.update(required_envs)

<<<<<<< HEAD
        tr0_var_1, tr1_var_1 = self._run_cluster(
            model_file, save_env, check_error_log, log_name=flag_name
        )
=======
        tr0_var_1, tr1_var_1 = self._run_cluster(model_file,
                                                 save_env,
                                                 check_error_log,
                                                 log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        load_env = {}
        load_env["LOAD"] = "1"
        load_env["MODEL_DIR"] = model_dir
        load_env.update(required_envs)
<<<<<<< HEAD
        tr0_var_2, tr1_var_2 = self._run_cluster(
            model_file, load_env, check_error_log, log_name=flag_name
        )
=======
        tr0_var_2, tr1_var_2 = self._run_cluster(model_file,
                                                 load_env,
                                                 check_error_log,
                                                 log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        shutil.rmtree(model_dir)

        train0_1_np = np.array(tr0_var_1)
        train1_1_np = np.array(tr1_var_1)
        train0_2_np = np.array(tr0_var_2)
        train1_2_np = np.array(tr1_var_2)

        np.testing.assert_almost_equal(train0_1_np, train0_2_np, decimal=2)
        np.testing.assert_almost_equal(train1_1_np, train1_2_np, decimal=2)

    def test_dist(self):
        need_envs = {
            "IS_DISTRIBUTED": '0',
            "IS_SPARSE": '0',
            'IS_SELF_CONTAINED_LR': '1',
            'SAVE_MODE': 'DIST',
            'OPTIMIZER': 'ADAM',
<<<<<<< HEAD
            'SKIP_STEPS': str(np.random.randint(2, 6)),
        }
        self.check_with_place(
            "dist_save_load.py",
            delta=0,
            check_error_log=True,
            need_envs=need_envs,
            log_name=flag_name,
        )
=======
            'SKIP_STEPS': str(np.random.randint(2, 6))
        }
        self.check_with_place("dist_save_load.py",
                              delta=0,
                              check_error_log=True,
                              need_envs=need_envs,
                              log_name=flag_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()

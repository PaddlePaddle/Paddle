# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os
import sys
import subprocess
import tempfile


class TestCollectiveAPIRunnerBase(object):

    def check_pass(self, *args, **kwargs):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def run_trainer(self, *args, **kwargs):
        self.check_pass(*args, **kwargs)


def runtime_main(test_class, col_type=None):
    args = {}
    model = test_class()
    args["static_mode"] = 0
    model.run_trainer(**args)


class TestDistBase(unittest.TestCase):

    def setUp(self):
        self._trainers = 4
        self._init_env()

    def _init_env(self):
        self._python_interp = sys.executable
        self.temp_dir = tempfile.TemporaryDirectory()

    def check_with_place(self,
                         model_file,
                         backend="nccl",
                         static_mode=False,
                         check_error_log=False,
                         need_envs={},
                         eager_mode=True,
                         args=[],
                         kwargs={}):
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "FLAGS_call_stack_level": "2",
            "GLOG_v": "0",
            "NCCL_P2P_DISABLE": "1",
            "PADDLE_WITH_GLOO": "0",
            "BACKEND": backend,
            "PADDLE_DISTRI_BACKEND": backend,
            "PADDLE_USE_GPU": "1"
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "0"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"

        if eager_mode:
            required_envs["FLAGS_enable_eager_mode"] = "%d" % 1
        else:
            required_envs["FLAGS_enable_eager_mode"] = "%d" % 0
        self._run_cluster(model_file, required_envs)

    def _run_cluster(self, model_file, envs):
        run_cluster_process = f"{self._python_interp} -u -m paddle.distributed.launch --log_dir {self.temp_dir.name} {model_file}"
        filted_envs = dict()
        for k in envs.keys():
            if "PADDLE_" == k[:7] and k not in [
                    "PADDLE_NNODES", "PADDLE_MASTER"
            ]:
                continue
            filted_envs[k] = envs[k]

        launcher = subprocess.Popen(run_cluster_process.strip().split(),
                                    stdout=sys.stderr,
                                    stderr=sys.stdout,
                                    env=filted_envs)
        launcher.communicate(timeout=240)

        if launcher.poll() is None:
            self.temp_dir.cleanup()
            raise TimeoutError
        elif launcher.poll() != 0:
            self.temp_dir.cleanup()
            raise RuntimeError("test failed!")
        self.temp_dir.cleanup()

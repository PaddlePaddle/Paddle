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

from __future__ import print_function
import numpy as np
import unittest
import time
import argparse
import os
import sys
import subprocess
import traceback
import functools
import pickle
import tempfile
from contextlib import closing
import paddle
import paddle.fluid as fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import socket


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
        self._ips = os.environ.get("PADDLE_TRAINERS", "127.0.0.1").split(",")
        self._trainer_id = int(os.environ.get("PADDLE_TRAINER_ID", 0))
        self._port_set = set()
        self._ps_endpoints = ""
        for i in range(self._trainers):
            self._ps_endpoints += f"{self._ips[i//len(self._ips)]}:{6010+i%8},"
        self._ps_endpoints = self._ps_endpoints[:-1]
        self._python_interp = sys.executable
        self.temp_dir = tempfile.TemporaryDirectory()

    def _find_free_port(self):

        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

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
            "GLOG_v": "3",
            "NCCL_P2P_DISABLE": "1",
            "PADDLE_WITH_GLOO": "0",
            "BACKEND": backend,
            "PADDLE_DISTRI_BACKEND": backend,
            "PADDLE_USE_GPU": "1"
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"

        if eager_mode:
            required_envs["FLAGS_enable_eager_mode"] = "%d" % 1
        else:
            required_envs["FLAGS_enable_eager_mode"] = "%d" % 0
        self._run_cluster(model_file, required_envs)

    def _run_cluster(self, model_file, envs):
        worker_endpoints = self._ps_endpoints.split(",")
        procs = []
        outputs = []
        tmppaths = []
        for i in range(self._trainers // len(self._ips)):
            endi = self._trainer_id * len(self._ips) + i
            env = {
                "FLAGS_selected_gpus": str(i),
                "PADDLE_TRAINER_ID": str(i),
                "PADDLE_TRAINERS_NUM": str(self._trainers),
                "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                "PADDLE_CURRENT_ENDPOINT": worker_endpoints[endi],
            }
            env.update(envs)
            out, stdpipe, path = self._run_cluster_one(model_file, env, i)
            procs.append(out)
            outputs.append(stdpipe)
            tmppaths.append(path)
        run_sucess = True
        for i, p in enumerate(procs):
            killed = False
            try:
                p.communicate(timeout=120)
                code = p.poll()
                if code is None:
                    raise subprocess.TimeoutExpired

            except subprocess.TimeoutExpired:
                p.kill()
                code = -15
                killed = True

            assert code is not None
            if p.returncode != 0:
                run_sucess = False
                if killed:
                    failed_type = "timed out"
                else:
                    failed_type = "failed"
                print(
                    f"========= subprocess {i} {failed_type}, following are the error messages(file:{tmppaths[i]}): ==========="
                )
                outputs[i].seek(0, 0)
                for line in outputs[i]:
                    print(line[:-1])
            outputs[i].close()
            #os.remove(tmppaths[i])

        if not run_sucess:
            raise RuntimeError("the process failed")

    def _run_cluster_one(self, model_file, envs, pi):
        #print("w0_ep:",w0_ep," w1_ep:",w1_ep)
        if core.is_compiled_with_cuda():
            env0 = {}
        else:
            raise NotImplementedError("Only support NCCL now")
        #update environment
        env0.update(envs)

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            tr_cmd = "%s -u -m coverage run --branch -p %s"
        else:
            tr_cmd = "%s -u %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        path0 = os.path.join(self.temp_dir.name,
                             f"/tmp/tr{pi}_err_%d.log" % os.getpid())
        tr0_pipe = open(path0, "w+")
        print(tr0_cmd)
        tr0_proc = subprocess.Popen(tr0_cmd.strip().split(),
                                    stdout=subprocess.PIPE,
                                    stderr=tr0_pipe,
                                    env=env0)
        return tr0_proc, tr0_pipe, path0

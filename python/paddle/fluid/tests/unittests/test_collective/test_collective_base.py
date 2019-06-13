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
import numpy as np
import unittest
import time
import argparse
import os
import six
import sys
import subprocess
import traceback
import functools
import pickle
import paddle.fluid as fluid


class TestCollectiveRunnerBase(object):
    def get_model(self, train_prog, startup_prog):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def initCommunicator(self, program, rank, wait_port, endpoints):
        raise NotImplementedError(
            "initCommunicator should be implemented by child class.")

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        self.initCommunicator(startup_prog, rank, nranks, True,
                              current_endpoint, endpoints)
        result = self.get_model(train_prog, startup_prog)
        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = fluid.CUDAPlace(
            device_id)  #if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        indata = np.arange(32).astype('int32').reshape((4, 8))
        out = exe.run(train_prog,
                      feed={'tindata': indata},
                      fetch_list=[result.name])
        if six.PY2:
            print(pickle.dumps(out))
        else:
            sys.stdout.buffer.write(out)


def runtime_main(test_class):
    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_gpus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    model.run_trainer(args)


import paddle.compat as cpt
import socket
from contextlib import closing


class TestDistBase(unittest.TestCase):
    def setUp(self):
        self._port_set = set()
        self._trainers = 2
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable

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

    def _run_cluster(self, model_file, envs):
        worker_endpoints = self._ps_endpoints.split(",")
        w0_ep, w1_ep = worker_endpoints
        #print("w0_ep:",w0_ep," w1_ep:",w1_ep)
        env0 = {
            "FLAGS_selected_gpus": "2",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep
        }

        env1 = {
            "FLAGS_selected_gpus": "3",
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w1_ep
        }
        #update environment
        env0.update(envs)
        env1.update(envs)
        tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        tr0_pipe = open("/tmp/tr0_err.log", "wb")
        tr1_pipe = open("/tmp/tr1_err.log", "wb")
        #print(tr0_cmd) 
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)

        tr1_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()
        sys.stderr.write('trainer 0 stderr: %s\n' % tr0_err)
        sys.stderr.write('trainer 1 stderr: %s\n' % tr1_err)
        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        return pickle.loads(tr0_out), pickle.loads(tr1_out)

    def check_with_place(self,
                         model_file,
                         col_type,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.99",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "GLOG_v": "0",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_IB_RETRY_CNT": "0",
        }
        a = np.arange(32).astype("int32")
        b = np.arange(32).astype("int32")
        c = a * 2
        d = np.append(a, b)
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
        tr0_out, tr1_out = self._run_cluster(model_file, required_envs)
        #print("@@@@@@@@@@@@@@@@:",tr0_out[0])
        #print("@@@@@@@@@@@@@@@@:",tr1_out[0])
        #testallgather
        if col_type == "allgather":
            self.assertTrue((tr0_out == d).all())
            self.assertTrue((tr1_out == d).all())
        elif col_type == "allreduce":
            pass
        elif col_type == "broadcast":
            pass
        elif col_type == "reduce_scatter":
            pass
        elif col_type == "reduce_scatter":
            pass
        else:
            pass

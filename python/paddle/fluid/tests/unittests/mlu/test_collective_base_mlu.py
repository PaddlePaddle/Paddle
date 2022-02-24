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
from contextlib import closing
import paddle.fluid as fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core


def DataTypeCast(date_type):
    np_data_type = None

    if date_type == "float16":
        np_data_type = np.float16
    elif date_type == "float32":
        np_data_type = np.float32
    elif date_type == "float64":
        np_data_type = np.float64
    elif date_type == "int8":
        np_data_type = np.int8
    elif date_type == "int16":
        np_data_type = np.int16
    elif date_type == "int32":
        np_data_type = np.int32
    elif date_type == "uint8":
        np_data_type = np.uint8
    else:
        raise ValueError("This data type is not support!")

    return np_data_type


class TestCollectiveRunnerBase(object):
    def get_model(self, train_prog, startup_prog):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def wait_server_ready(self, endpoints):
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                        socket.socket(socket.AF_INET,
                                      socket.SOCK_STREAM)) as sock:
                    sock.settimeout(2)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if hasattr(socket, 'SO_REUSEPORT'):
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT,
                                        1)

                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write("not ready endpoints:" + str(
                    not_ready_endpoints) + "\n")
                sys.stderr.flush()
                time.sleep(3)
            else:
                break

#endpoints should be ["ip1:port1","ip2:port2"]

    def initCommunicator(self, program, rank, nranks, wait_port,
                         current_endpoint, endpoints):
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            self.wait_server_ready(other_endpoints)
        block = program.global_block()
        cncl_id_var = block.create_var(
            name=nameGen.generate('cncl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)

        block.append_op(
            type='c_gen_cncl_id',
            inputs={},
            outputs={'Out': cncl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints
            })

        block.append_op(
            type='c_comm_init',
            inputs={'X': cncl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': self.global_ring_id
            })

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        self.initCommunicator(startup_prog, rank, nranks, True,
                              current_endpoint, endpoints)
        self.rank = rank
        result = self.get_model(train_prog, startup_prog)
        device_id = int(os.getenv("FLAGS_selected_mlus", "0"))
        place = fluid.MLUPlace(device_id)
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        np.random.seed(os.getpid())
        np_data_type = DataTypeCast(args["data_type"])
        indata = np.random.random((10, 1000)).astype(np_data_type)
        out = exe.run(train_prog,
                      feed={'tindata': indata},
                      fetch_list=[result.name])
        sys.stdout.buffer.write(pickle.dumps(out))


def runtime_main(test_class, col_type, sub_type):
    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_mlus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
    args["data_type"] = os.getenv("DATA_TYPE")
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
            "FLAGS_selected_mlus": "0",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep
        }

        env1 = {
            "FLAGS_selected_mlus": "1",
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
        return pickle.loads(tr0_out), pickle.loads(
            tr1_out), tr0_proc.pid, tr1_proc.pid

    def check_with_place(self,
                         model_file,
                         col_type,
                         data_type,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "GLOG_v": "3",
            "DATA_TYPE": data_type,
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(model_file,
                                                         required_envs)
        np_data_type = DataTypeCast(data_type)
        np.random.seed(pid0)
        input1 = np.random.random((10, 1000)).astype(np_data_type)
        np.random.seed(pid1)
        input2 = np.random.random((10, 1000)).astype(np_data_type)
        if col_type == "broadcast":
            need_result = input2
            self.assertTrue(np.allclose(tr0_out, need_result))
            self.assertTrue(np.allclose(tr1_out, need_result))
        elif col_type == "allreduce":
            need_result = input1 + input2
            self.assertTrue(
                np.allclose(
                    tr0_out, need_result, rtol=1e-05, atol=1e-05))
            self.assertTrue(
                np.allclose(
                    tr1_out, need_result, rtol=1e-05, atol=1e-05))
        else:
            pass

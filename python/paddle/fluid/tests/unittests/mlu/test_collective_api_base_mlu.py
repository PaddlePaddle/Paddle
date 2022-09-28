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

import numpy as np
import unittest
import os
import sys
import subprocess
import pickle
from contextlib import closing
import paddle
import paddle.fluid as fluid
from paddle.fluid import core


def DataTypeCast(date_type):
    np_data_type = None

    if date_type == "float16":
        np_data_type = np.float16
    elif date_type == "float32":
        np_data_type = np.float32
    elif date_type == "int32":
        np_data_type = np.int32
    else:
        raise ValueError("This data type is not support!")

    return np_data_type


class TestCollectiveAPIRunnerBase(object):

    def get_model(self, train_prog, startup_prog, rank, indata=None):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        paddle.distributed.init_parallel_env()
        device_id = int(os.getenv("FLAGS_selected_mlus", "0"))
        place = fluid.MLUPlace(device_id)
        np.random.seed(os.getpid())
        np_data_type = DataTypeCast(args["data_type"])
        indata = np.random.random((10, 1000)).astype(np_data_type)
        if args['static_mode']:
            result = self.get_model(train_prog, startup_prog, rank)
            exe = fluid.Executor(place)
            exe.run(startup_prog)
            fetch_list = []
            for elem in result:
                fetch_list.append(elem.name)
            out = exe.run(train_prog,
                          feed={'tindata': indata},
                          fetch_list=fetch_list)
        else:
            out = self.get_model(train_prog, startup_prog, rank, indata)
            #print(out, sys.stderr)
        sys.stdout.buffer.write(pickle.dumps(out))


def runtime_main(test_class, col_type):
    args = {}
    model = test_class()
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
    args["backend"] = os.getenv("BACKEND")
    args["path_id"] = int(os.getenv("PATH_ID"))
    args["static_mode"] = int(os.getenv("STATIC_MODE"))
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
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            tr_cmd = "%s -m coverage run --branch -p %s"
        else:
            tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        tr0_pipe = open("/tmp/tr0_err_%d.log" % os.getpid(), "w")
        tr1_pipe = open("/tmp/tr1_err_%d.log" % os.getpid(), "w")
        #print(tr0_cmd)
        tr0_proc = subprocess.Popen(tr0_cmd.strip().split(),
                                    stdout=subprocess.PIPE,
                                    stderr=tr0_pipe,
                                    env=env0)

        tr1_proc = subprocess.Popen(tr0_cmd.strip().split(),
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
        with open("/tmp/tr0_err_%d.log" % os.getpid(), "r") as f:
            sys.stderr.write('trainer 0 stderr file: %s\n' % f.read())
        with open("/tmp/tr1_err_%d.log" % os.getpid(), "r") as f:
            sys.stderr.write('trainer 1 stderr file: %s\n' % f.read())
        return pickle.loads(tr0_out), pickle.loads(
            tr1_out), tr0_proc.pid, tr1_proc.pid

    def check_with_place(self,
                         model_file,
                         col_type,
                         data_type,
                         path_id="0",
                         static_mode="1",
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "FLAGS_call_stack_level": "2",
            "GLOG_v": "3",
            "STATIC_MODE": static_mode,
            "PADDLE_WITH_GLOO": '0',
            "BACKEND": "cncl",
            "PATH_ID": path_id,
            "DATA_TYPE": data_type
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(
            model_file, required_envs)
        np_data_type = DataTypeCast(data_type)
        np.random.seed(pid0)
        input1 = np.random.random((10, 1000)).astype(np_data_type)
        np.random.seed(pid1)
        input2 = np.random.random((10, 1000)).astype(np_data_type)
        if col_type == "broadcast":
            need_result = input2
            np.testing.assert_allclose(tr0_out, need_result)
            np.testing.assert_allclose(tr1_out, need_result)
        elif col_type == "allreduce":
            need_result = input1 + input2
            np.testing.assert_allclose(tr0_out,
                                       need_result,
                                       rtol=1e-05,
                                       atol=1e-05)
            np.testing.assert_allclose(tr1_out,
                                       need_result,
                                       rtol=1e-05,
                                       atol=1e-05)
        elif col_type == "reduce":
            need_result = input1 + input2
            np.testing.assert_allclose(tr0_out, need_result)
        elif col_type == "allgather":
            need_result = np.vstack((input1, input2))
            tr_out0 = np.vstack((tr0_out[0], tr0_out[1]))
            tr_out1 = np.vstack((tr1_out[0], tr1_out[1]))
            np.testing.assert_allclose(tr_out0, need_result)
            np.testing.assert_allclose(tr_out1, need_result)
        else:
            pass

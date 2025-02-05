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

import os
import pickle
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing

import numpy as np

import paddle.base.unique_name as nameGen
from paddle import base
from paddle.base import core
from paddle.distributed.collective import _init_parallel_env


def DataTypeCast(date_type):
    np_dtype = None

    if date_type == "float16":
        np_dtype = np.float16
    elif date_type == "float32":
        np_dtype = np.float32
    elif date_type == "float64":
        np_dtype = np.float64
    elif date_type == "uint8":
        np_dtype = np.uint8
    elif date_type == "int32":
        np_dtype = np.int32
    elif date_type == "int64":
        np_dtype = np.int64
    elif date_type == "bfloat16":
        np_dtype = np.uint16
    else:
        raise ValueError("This data type is not support!")

    return np_dtype


def dump_output(x):
    dump_file = os.environ['DUMP_FILE']
    with open(dump_file, 'wb') as f:
        pickle.dump(x, f)


class TestCollectiveRunnerBase:
    def get_model(self, train_prog, startup_prog, dtype=None):
        raise NotImplementedError(
            "get model should be implemented by child class."
        )

    def wait_server_ready(self, endpoints):
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ) as sock:
                    sock.settimeout(2)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if hasattr(socket, 'SO_REUSEPORT'):
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
                        )

                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write(
                    "not ready endpoints:" + str(not_ready_endpoints) + "\n"
                )
                sys.stderr.flush()
                time.sleep(3)
            else:
                break

    # endpoints should be ["ip1:port1","ip2:port2"]

    def initCommunicator(
        self, program, rank, nranks, wait_port, current_endpoint, endpoints
    ):
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            self.wait_server_ready(other_endpoints)
        block = program.global_block()
        bkcl_id_var = block.create_var(
            name=nameGen.generate('bkcl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW,
        )

        block.append_op(
            type='c_gen_bkcl_id',
            inputs={},
            outputs={'Out': bkcl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
            },
        )

        block.append_op(
            type='c_comm_init',
            inputs={'X': bkcl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': self.global_ring_id,
            },
        )

    def run_trainer(self, args):
        train_prog = base.Program()
        startup_prog = base.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2

        _init_parallel_env("bkcl")

        self.rank = rank
        np_dtype = DataTypeCast(args["dtype"])
        result = self.get_model(train_prog, startup_prog, np_dtype)
        device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
        place = base.XPUPlace(device_id)
        exe = base.Executor(place)
        exe.run(startup_prog)
        np.random.seed(os.getpid())
        indata = np.random.uniform(
            low=-10.0, high=10.0, size=(10, 1000)
        ).astype(np_dtype)
        out = exe.run(
            train_prog, feed={'tindata': indata}, fetch_list=[result.name]
        )
        dump_output(out[0])


def runtime_main(test_class, col_type, sub_type):
    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_xpus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
    args["dtype"] = os.getenv("DTYPE")
    args["batch_size"] = os.getenv("BATCH_SIZE")
    model.run_trainer(args)


class TestDistBase(unittest.TestCase):
    def setUp(self):
        self._port_set = set()
        self._trainers = 2
        self._ps_endpoints = f"127.0.0.1:{self._find_free_port()},127.0.0.1:{self._find_free_port()}"
        self._python_interp = sys.executable

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _find_free_port(self):
        def __free_port():
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
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
        env0 = {
            "FLAGS_selected_xpus": "0",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep,
        }

        env1 = {
            "FLAGS_selected_xpus": "1",
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w1_ep,
        }
        # update environment
        env0.update(envs)
        env1.update(envs)

        # setup out dump path
        cur_pid = os.getpid()
        dump_file_0 = f'./out_data_0_{cur_pid}.pickled'
        dump_file_1 = f'./out_data_1_{cur_pid}.pickled'
        env0['DUMP_FILE'] = dump_file_0
        env1['DUMP_FILE'] = dump_file_1

        tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        path0 = os.path.join(self.temp_dir.name, "/tmp/tr0_err.log")
        path1 = os.path.join(self.temp_dir.name, "/tmp/tr1_err.log")
        tr0_pipe = open(path0, "wb")
        tr1_pipe = open(path1, "wb")
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            # stderr=tr0_pipe,
            env=env0,
        )

        tr1_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            # stderr=tr1_pipe,
            env=env1,
        )

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()
        sys.stderr.write(f'trainer 0 stderr: {tr0_err}\n')
        sys.stderr.write(f'trainer 1 stderr: {tr1_err}\n')
        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()

        def load_and_remove(path):
            with open(path, 'rb') as f:
                out = pickle.load(f)
            os.remove(path)
            return out

        return (
            load_and_remove(dump_file_0),
            load_and_remove(dump_file_1),
            tr0_proc.pid,
            tr1_proc.pid,
        )

    def check_with_place(
        self,
        model_file,
        col_type,
        dtype=None,
        check_error_log=False,
        need_envs={},
    ):
        required_envs = {
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "GLOG_v": "3",
            "DTYPE": dtype,
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(
            model_file, required_envs
        )
        dtype = "float32" if dtype is None else dtype
        np_dtype = DataTypeCast(dtype)
        np.random.seed(pid0)
        input1 = np.random.uniform(
            low=-10.0, high=10.0, size=(10, 1000)
        ).astype(np_dtype)
        np.random.seed(pid1)
        input2 = np.random.uniform(
            low=-10.0, high=10.0, size=(10, 1000)
        ).astype(np_dtype)
        if col_type == "allgather":
            need_result = np.vstack((input1, input2))
            np.testing.assert_allclose(tr0_out, need_result)
            np.testing.assert_allclose(tr1_out, need_result)
        elif col_type == "broadcast":
            need_result = input2
            np.testing.assert_allclose(tr0_out, need_result)
            np.testing.assert_allclose(tr1_out, need_result)
        elif col_type == "reduce":
            need_result = input1 + input2
            np.testing.assert_allclose(tr1_out, need_result)
        elif col_type == "scatter":
            need_result = input2
            need_result1 = need_result[0 : need_result.shape[0] // 2]
            need_result2 = need_result[need_result.shape[0] // 2 :]
            np.testing.assert_allclose(tr0_out, need_result1)
            np.testing.assert_allclose(tr1_out, need_result2)
        elif col_type == "allreduce":
            need_result = input1 + input2
            np.testing.assert_allclose(
                tr0_out, need_result, rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                tr1_out, need_result, rtol=1e-05, atol=1e-05
            )
        elif col_type == "reduce_scatter":
            tmp = input1 + input2
            need_result1 = tmp[0 : tmp.shape[0] // 2]
            need_result2 = tmp[tmp.shape[0] // 2 :]
            np.testing.assert_allclose(
                tr0_out, need_result1, rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                tr1_out, need_result2, rtol=1e-05, atol=1e-05
            )
        elif col_type == "sendrecv":
            need_result = input1
            np.testing.assert_allclose(
                tr1_out, need_result, rtol=1e-05, atol=1e-05
            )
        elif col_type == "identity":
            need_result1 = input1
            need_result2 = input2
            np.testing.assert_allclose(tr0_out, need_result1, rtol=0, atol=0)
            np.testing.assert_allclose(tr1_out, need_result2, rtol=0, atol=0)
        elif col_type == "reduce_slicegather":
            slice_size = input1.shape[0] // 2
            tmp10 = input1[0:slice_size]
            tmp11 = input2[0:slice_size]
            need_result1 = np.concatenate((tmp10, tmp11), axis=1)
            tmp20 = input1[slice_size:]
            tmp21 = input2[slice_size:]
            need_result2 = np.concatenate((tmp20, tmp21), axis=1)
            np.testing.assert_allclose(tr0_out, need_result1)
            np.testing.assert_allclose(tr1_out, need_result2)
        elif col_type == "concat":
            need_result = np.concatenate((input1, input2), axis=1)
            np.testing.assert_allclose(
                tr0_out, need_result, rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                tr1_out, need_result, rtol=1e-05, atol=1e-05
            )
        elif col_type == "split":
            need_result1 = np.split(input1, 2, axis=1)[0]
            need_result2 = np.split(input2, 2, axis=1)[1]
            np.testing.assert_allclose(
                tr0_out, need_result1, rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                tr1_out, need_result2, rtol=1e-05, atol=1e-05
            )
        elif col_type == "sendrecv_array":
            need_result1 = np.array([[0, 1, 2]])
            need_result2 = np.array([[3, 4, 5]])
            np.testing.assert_allclose(
                tr1_out[0][0], need_result1, rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                tr1_out[0][1], need_result2, rtol=1e-05, atol=1e-05
            )
        else:
            pass

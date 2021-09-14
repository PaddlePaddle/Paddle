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
import sys
sys.path.append("..")
import subprocess
import traceback
import functools
import pickle
from contextlib import closing
import paddle.fluid as fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
from six import string_types
import paddle

from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator

_set_use_system_allocator(False)
paddle.enable_static()


def create_or_get_tensor(scope, var_name, var, place):
    print('test_sync_batch_norm_base_npu.py create_or_get_tensor')
    print('var_name: ', var_name)
    # print('var: ', var)
    print('place: ', place)
    """Get tensor, if not found, create a new one."""
    tensor = scope.var(var_name).get_tensor()
    # print('tensor: ', tensor)
    print('create_or_get_tensor 41')
    # with paddle.static.device_guard('NPUPlace(0)'):
    # with paddle.static.device_guard(place):
    with paddle.static.device_guard('npu'):
        if var is not None:
            print('create_or_get_tensor 43')
            assert isinstance(var, np.ndarray)
            print('create_or_get_tensor 44')
            tensor.set_recursive_sequence_lengths([])
            print('create_or_get_tensor 47')
            tensor.set(var, place)
            print('create_or_get_tensor 49')
        print('create_or_get_tensor 50')
    return tensor


class TestSyncBatchNormRunnerBase(object):
    # def get_model(self, train_prog, startup_prog):
    def get_model(self,
                  main,
                  startup,
                  place,
                  layout,
                  seed,
                  sync_bn=False,
                  only_forward=False):
        raise NotImplementedError(
            "get model should be implemented by child class.")

    def wait_server_ready(self, endpoints):
        assert not isinstance(endpoints, string_types)
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                        socket.socket(socket.AF_INET,
                                      socket.SOCK_STREAM)) as sock:
                    sock.settimeout(2)
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
        hccl_id_var = block.create_var(
            name=nameGen.generate('hccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        block.append_op(
            type='c_gen_hccl_id',
            inputs={},
            outputs={'Out': hccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints
            })
        block.append_op(
            type='c_comm_init_hccl',
            inputs={'X': hccl_id_var},
            outputs={},
            attrs={
                'rank': rank,
                'ring_id': self.global_ring_id,
                'device_id': int(os.getenv("FLAGS_selected_npus")),
                'rank_ids': nranks
            })

    def run_trainer(self, args):
        print('test_sync_batch_norm_base_npu.py run_trainer')

        device_id = int(os.getenv("FLAGS_selected_npus", "0"))
        print("use selected_npus:", device_id)
        place = fluid.NPUPlace(device_id)

        print(' run_trainer 124 ')
        seed = 10
        scope = core.Scope()
        print(' run_trainer 127 ')
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        print(' data: ', data)
        print('run_trainer 129')
        data = create_or_get_tensor(scope, "input",
                                    OpTest.np_dtype_to_fluid_dtype(data), place)
        print(' data: ', data)

        print(' run_trainer 135')

        train_prog = fluid.Program()
        startup_prog = fluid.Program()

        # print('train_prog: ', train_prog)
        # print('startup_prog: ', startup_prog)

        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        self.initCommunicator(startup_prog, rank, nranks, True,
                              current_endpoint, endpoints)
        # self.initCommunicator(train_prog, rank, nranks, True,
        #                       current_endpoint, endpoints)
        # print('startup_prog: ', startup_prog)
        self.rank = rank

        outs = self.get_model(train_prog, startup_prog, place, "NCHW", seed)

        # print('before train_prog: ', train_prog)
        # print(' ', train_prog.blocks[0].ops[3], '')
        # train_prog.blocks[0].ops[3].desc.set_type('sync_batch_norm')
        # train_prog.blocks[0].ops[9].desc.set_type('sync_batch_norm_grad')
        # print('after train_prog: ', train_prog)
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        only_forward = False
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_3@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=train_prog,
                             feed={'input': data},
                             fetch_list=fetch_names)
        # sys.stdout.buffer.write(pickle.dumps(bn_fetches))


def runtime_main(test_class, col_type, sub_type):
    print('test_sync_batch_norm_base_npu.py runtime_main')

    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_npus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
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
        print(' test_sync_batch_norm_base_npu.py _run_cluster ')
        # print(' envs: ', envs)

        worker_endpoints = self._ps_endpoints.split(",")
        w0_ep, w1_ep = worker_endpoints
        print("w0_ep:", w0_ep, " w1_ep:", w1_ep)
        env0 = {
            "FLAGS_selected_npus": "0",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep,
        }

        env1 = {
            "FLAGS_selected_npus": "1",
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w1_ep,
        }
        #update environment
        env0.update(envs)
        env1.update(envs)
        # print(' env0: ', env0)
        # print(' env1: ', env1)

        tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        tr0_pipe = open("/tmp/tr0_err.log", "wb")
        tr1_pipe = open("/tmp/tr1_err.log", "wb")
        print(tr0_cmd)
        # print(tr1_cmd) 
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

        print('tr0_out: ', tr0_out)
        # print('tr1_out: ', tr1_out)
        # print('tr0_err: ', tr0_err)
        # print('tr1_err: ', tr1_err)

        sys.stderr.write('trainer 0 stderr: %s\n' % tr0_err)
        sys.stderr.write('trainer 1 stderr: %s\n' % tr1_err)
        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        # return pickle.loads(tr0_out), pickle.loads(
        #     tr1_out), tr0_proc.pid, tr1_proc.pid

    def check_with_place(self, model_file, col_type, need_envs={}):
        print('test_sync_batch_norm_base_npu.py check_with_place')

        self._run_cluster(model_file, need_envs)

        # tr0_out, tr1_out, pid0, pid1 = self._run_cluster(model_file, need_envs)
        # np.random.seed(pid0)
        # input1 = np.random.random((10, 1000))
        # np.random.seed(pid1)
        # input2 = np.random.random((10, 1000))
        # if col_type == "identity":
        #     need_result1 = input1
        #     need_result2 = input2
        #     self.assertTrue(np.allclose(tr0_out, need_result1, rtol=0, atol=0))
        #     self.assertTrue(np.allclose(tr1_out, need_result2, rtol=0, atol=0))

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

import os
import signal
import time
import shutil
import unittest

from multiprocessing import Process

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import Program, program_guard
from dist_test_utils import *
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode


def run_pserver(pserver_id):
    remove_ps_flag(os.getpid())
    scope = fluid.core.Scope()
    program = Program()
    with fluid.scope_guard(scope):
        with program_guard(program, startup_program=Program()):
            # create table parameter in scope
            place = fluid.CPUPlace()
            # create and initialize Param Variable
            param = scope.var('table').get_tensor()

            param_array = np.ones((5, 8)).astype("float32")
            for i in range(len(param_array)):
                param_array[i] *= param_array[i] * i + pserver_id * 10 + 1
            param.set(param_array, place)

            optimize_block = program._create_block(program.global_block().idx)
            program.global_block().append_op(type="listen_and_serv",
                                             inputs={'X': []},
                                             outputs={},
                                             attrs={
                                                 "optimize_blocks":
                                                 [optimize_block],
                                                 "endpoint":
                                                 '127.0.0.1:0',
                                                 "Fanin":
                                                 1,
                                                 "distributed_mode":
                                                 DistributedMode.SYNC,
                                                 "grad_to_block_id": []
                                             })

            exe = fluid.Executor(place)
            exe.run(program)


@unittest.skip("do not need currently")
class TestListenAndServOp(unittest.TestCase):

    def setUp(self):
        self.ps_timeout = 5

    def _start_pserver(self, pserver_id, pserver_func):
        p = Process(target=pserver_func, args=(pserver_id, ))
        p.daemon = True
        p.start()
        return p

    def _wait_ps_ready(self, pid):
        start_left_time = self.ps_timeout
        sleep_time = 0.5
        while True:
            assert start_left_time >= 0, "wait ps ready failed"
            time.sleep(sleep_time)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                start_left_time -= sleep_time

    def _get_pserver_port(self, pid):
        with open("/tmp/paddle.%d.port" % pid, 'r') as f:
            port = int(f.read().strip())
        return port

    def _run_nce_op_two_pserver(self, place, port0, port1, model_file):
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                emaps = ['127.0.0.1:' + str(port0), '127.0.0.1:' + str(port1)]

                # create and run recv and save operator
                remote_recv_op = Operator("recv_save",
                                          trainer_id=0,
                                          shape=[10, 8],
                                          slice_shapes=["5,8", "5,8"],
                                          slice_varnames=["table", "table"],
                                          remote_varnames=['table', 'table'],
                                          is_sparse=False,
                                          endpoints=emaps,
                                          file_path=model_file)

                remote_recv_op.run(scope, place)

    def _load_slice_var(self, model_file):
        load_prog = fluid.Program()
        load_block = load_prog.global_block()

        origin = load_block.create_var(
            name="var.origin",
            type=fluid.core.VarDesc.VarType.LOD_TENSOR,
            shape=[10, 8],
            dtype="float32",
            persistable=True)

        slice0 = load_block.create_var(
            name="var.slice0",
            type=fluid.core.VarDesc.VarType.LOD_TENSOR,
            shape=[3, 8],
            dtype="float32",
            persistable=True)

        slice1 = load_block.create_var(
            name="var.slice1",
            type=fluid.core.VarDesc.VarType.LOD_TENSOR,
            shape=[5, 8],
            dtype="float32",
            persistable=True)

        load_block.append_op(type='load',
                             inputs={},
                             outputs={'Out': [origin]},
                             attrs={'file_path': model_file})

        load_block.append_op(type='load',
                             inputs={},
                             outputs={'Out': [slice0]},
                             attrs={
                                 'file_path': model_file,
                                 'seek': 2 * 8,
                                 'shape': slice0.shape
                             })

        load_block.append_op(type='load',
                             inputs={},
                             outputs={'Out': [slice1]},
                             attrs={
                                 'file_path': model_file,
                                 'seek': 5 * 8,
                                 'shape': slice1.shape
                             })

        exe = fluid.Executor(place=fluid.CPUPlace())
        exe.run(load_prog)

        origin_var = fluid.global_scope().find_var("var.origin")
        slice0_var = fluid.global_scope().find_var("var.slice0")
        slice1_var = fluid.global_scope().find_var("var.slice1")

        origin = np.array(origin_var.get_tensor())
        slice0 = np.array(slice0_var.get_tensor())
        slice1 = np.array(slice1_var.get_tensor())

        np.testing.assert_equal(origin[2:5], slice0)
        np.testing.assert_equal(origin[5:10], slice1)

    def _save_by_io_persistables(self, place, port0, port1, dirname, var_name):
        self._run_nce_op_two_pserver(place, port0, port1,
                                     os.path.join(dirname, var_name))

    def test_recv_save_op_remote(self):
        # run pserver on CPU in sync mode
        p0 = self._start_pserver(0, run_pserver)
        self._wait_ps_ready(p0.pid)
        port0 = self._get_pserver_port(p0.pid)

        p1 = self._start_pserver(1, run_pserver)
        self._wait_ps_ready(p1.pid)
        port1 = self._get_pserver_port(p1.pid)

        places = [core.CPUPlace()]

        param_dir = "./model_for_test_recv_save_op/"
        param_name = "table"

        for place in places:
            self._save_by_io_persistables(place, port0, port1, param_dir,
                                          param_name)

        # raise SIGTERM to pserver
        os.kill(p0.pid, signal.SIGINT)
        p0.join()
        os.kill(p1.pid, signal.SIGINT)
        p1.join()

        self._load_slice_var(param_dir + param_name)
        shutil.rmtree(param_dir)


if __name__ == '__main__':
    unittest.main()

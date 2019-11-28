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
import signal
import time
import unittest
from multiprocessing import Process

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import Program, program_guard
from dist_test_utils import *

# 
# load_prog = fluid.Program()
# load_block = load_prog.global_block()
#
# origin = load_block.create_var(
#     name="{}.load".format("var"),
#     type=fluid.core.VarDesc.VarType.LOD_TENSOR,
#     shape=[10,8],
#     dtype="float32",
#     persistable=True)
#
# load_block.append_op(
#     type='load',
#     inputs={},
#     outputs={'Out': [origin]},
#     attrs={'file_path': "xxxxx.fc.files"})
#
# exe = fluid.Executor(place=fluid.CPUPlace())
# exe.run(load_prog)
#
# origin_var = fluid.global_scope().find_var("var.load")
#
# print(np.array(origin_var.get_tensor()))


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
            program.global_block().append_op(
                type="listen_and_serv",
                inputs={'X': []},
                outputs={},
                attrs={
                    "optimize_blocks": [optimize_block],
                    "endpoint": '127.0.0.1:0',
                    "Fanin": 1,
                    "sync_mode": True,
                    "grad_to_block_id": []
                })

            exe = fluid.Executor(place)
            exe.run(program)


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

    def _run_nce_op_two_pserver(self, place, port0, port1):
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                emaps = ['127.0.0.1:' + str(port0), '127.0.0.1:' + str(port1)]

                # create and run recv and save operator
                remote_recv_op = Operator(
                    "recv_save",
                    trainer_id=0,
                    shape=[10, 8],
                    slice_shapes=["5,8", "5,8"],
                    slice_varnames=["table", "table"],
                    remote_varnames=['table', 'table'],
                    endpoints=emaps,
                    file_path="./xxxxx.fc.files")

                remote_recv_op.run(scope, place)

    def test_nce_op_remote(self):
        # run pserver on CPU in sync mode
        p0 = self._start_pserver(0, run_pserver)
        self._wait_ps_ready(p0.pid)
        port0 = self._get_pserver_port(p0.pid)

        p1 = self._start_pserver(1, run_pserver)
        self._wait_ps_ready(p1.pid)
        port1 = self._get_pserver_port(p1.pid)

        places = [core.CPUPlace()]

        for place in places:
            self._run_nce_op_two_pserver(place, port0, port1)

        # raise SIGTERM to pserver
        os.kill(p0.pid, signal.SIGINT)
        p0.join()
        os.kill(p1.pid, signal.SIGINT)
        p1.join()


if __name__ == '__main__':
    unittest.main()

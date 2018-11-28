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


def run_pserver(use_cuda, sync_mode):
    scope = fluid.core.Scope()
    program = Program()
    with fluid.scope_guard(scope):
        with program_guard(program, startup_program=Program()):
            # create table parameter in scope
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            # create and initialize Param Variable
            param = scope.var('table').get_tensor()
            param_array = np.full((10, 8), 5.0).astype("float32")
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

    def _start_pserver(self, use_cuda, sync_mode, pserver_func):
        p = Process(target=pserver_func, args=(use_cuda, sync_mode))
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

    def _run_lookup_table_op(self, place, port):
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                # create and initialize Param Variable
                param = scope.var('W').get_tensor()
                param_array = np.full((10, 8), 1.0).astype("float32")
                param.set(param_array, place)

                ids = scope.var('Ids').get_tensor()
                ids_array = np.array([[1.0], [2.0]]).astype("int64")
                ids.set(ids_array, place)
                ids.set_lod([[0, 1, 2]])

                out = scope.var('Out').get_tensor()

                emaps = ['127.0.0.1:' + str(port)]
                table_names = ['table']
                height_sections = [10]
                # create and run sgd operator
                lookup_table_op = Operator(
                    "lookup_table",
                    W='W',
                    Ids='Ids',
                    Out='Out',
                    remote_prefetch=True,
                    epmap=emaps,
                    table_names=table_names,
                    height_sections=height_sections)
                lookup_table_op.run(scope, place)

                # get and compare result
                result_array = np.array(out)

                print(result_array)

                self.assertTrue((result_array[0] == 5).all())
                self.assertTrue((result_array[0] == 5).all())

    def test_lookup_remote_table(self):
        # run pserver on CPU in sync mode
        p1 = self._start_pserver(False, True, run_pserver)
        self._wait_ps_ready(p1.pid)
        port = self._get_pserver_port(p1.pid)

        places = [core.CPUPlace()]
        # if core.is_compiled_with_cuda():
        #     places.append(core.CUDAPlace(0))
        for place in places:
            self._run_lookup_table_op(place, port)

        # raise SIGTERM to pserver
        os.kill(p1.pid, signal.SIGINT)
        p1.join()


if __name__ == '__main__':
    unittest.main()

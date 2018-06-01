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

import paddle
import paddle.fluid as fluid
import os
import signal
import subprocess
import time
import unittest
from multiprocessing import Process
from op_test import OpTest


def run_pserver(use_cuda, sync_mode, ip, port, trainers, trainer_id):
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # loss function
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    # optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    pserver_endpoints = ip + ":" + port
    current_endpoint = ip + ":" + port
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=sync_mode)
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)


class TestListenAndServOp(OpTest):
    def setUp(self):
        self.ps_timeout = 5
        self.ip = "127.0.0.1"
        self.port = "6173"
        self.trainers = 1
        self.trainer_id = 1

    def _start_pserver(self, use_cuda, sync_mode):
        p = Process(
            target=run_pserver,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainers,
                  self.trainer_id))
        p.start()
        return p.pid

    def _wait_ps_ready(self, pid):
        retry_times = self.ps_timeout
        while True:
            time.sleep(1)
            assert retry_times >= 0, "wait ps ready failed"
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def test_handle_signal_in_serv_op(self):
        # run pserver on CPU in sync mode
        pid = self._start_pserver(False, True)
        self._wait_ps_ready(pid)

        # raise SIGINT to pserver
        os.kill(pid, signal.SIGINT)

        # run pserver on CPU in async mode
        pid = self._start_pserver(False, False)
        self._wait_ps_ready(pid)

        # raise SIGTERM to pserver
        os.kill(pid, signal.SIGINT)


if __name__ == '__main__':
    unittest.main()

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


def run_pserver(use_cuda, sync_mode, ip, port, trainer_count, trainer_id):
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

    port = os.getenv("PADDLE_INIT_PORT", port)
    pserver_ips = os.getenv("PADDLE_INIT_PSERVERS", ip)  # ip,ip...
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
    trainers = int(os.getenv("TRAINERS", trainer_count))
    current_endpoint = os.getenv("POD_IP", ip) + ":" + port
    trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID", trainer_id))
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
        self.sleep_time = 5
        self.ip = "127.0.0.1"
        self.port = "6173"
        self.trainer_count = 1
        self.trainer_id = 1

    def _raise_signal(self, parent_pid, raised_signal):
        time.sleep(self.sleep_time)
        ps_command = subprocess.Popen(
            "ps -o pid --ppid %d --noheaders" % parent_pid,
            shell=True,
            stdout=subprocess.PIPE)
        ps_output = ps_command.stdout.read()
        retcode = ps_command.wait()
        assert retcode == 0, "ps command returned %d" % retcode

        for pid_str in ps_output.split("\n")[:-1]:
            try:
                os.kill(int(pid_str), raised_signal)
            except Exception:
                continue

    def _start_pserver(self, use_cuda, sync_mode):
        p = Process(
            target=run_pserver,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainer_count,
                  self.trainer_id))
        p.start()

    def test_handle_signal_in_serv_op(self):
        # run pserver on CPU in sync mode
        self._start_pserver(False, True)

        # raise SIGINT to pserver
        self._raise_signal(os.getpid(), signal.SIGINT)

        # run pserver on CPU in async mode
        self._start_pserver(False, False)

        # raise SIGTERM to pserver
        self._raise_signal(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()

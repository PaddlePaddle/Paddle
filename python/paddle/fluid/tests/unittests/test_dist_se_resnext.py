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

import numpy as np
import argparse
import time
import math

import unittest
import os
import signal
import subprocess


class TestDistSeResneXt2x2(unittest.TestCase):
    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._ps_endpoints = "127.0.0.1:9123,127.0.0.1:9124"
        self._python_interp = "/opt/python/cp27-cp27mu/bin/python"

    def start_pserver(self):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps0_cmd = "%s dist_se_resnext.py pserver %s 0 %s %d" % \
            (self._python_interp, self._ps_endpoints, ps0_ep, self._trainers)
        ps1_cmd = "%s dist_se_resnext.py pserver %s 0 %s %d" % \
            (self._python_interp, self._ps_endpoints, ps1_ep, self._trainers)

        ps0_proc = subprocess.Popen(ps0_cmd.split(" "), stdout=subprocess.PIPE)
        ps1_proc = subprocess.Popen(ps1_cmd.split(" "), stdout=subprocess.PIPE)
        return ps0_proc, ps1_proc

    def _wait_ps_ready(self, pid):
        retry_times = 20
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            print("waiting ps ready: ", pid)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def test_with_place(self):

        ps0, ps1 = self.start_pserver()
        self._wait_ps_ready(ps0.pid)
        self._wait_ps_ready(ps1.pid)
        print("wait end")

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        tr0_cmd = "%s dist_se_resnext.py trainer %s 0 %s %d" % \
            (self._ps_endpoints, ps0_ep, self._trainers)
        tr1_cmd = "%s dist_se_resnext.py pserver %s 1 %s %d" % \
            (self._python_interp, self._ps_endpoints, ps1_ep, self._trainers)

        tr0_proc = subprocess.Popen(
            tr0_cmd.split(" "),
            stdout=subprocess.PIPE,
            env={"CUDA_VISIBLE_DEVICES": "0,1"})
        tr1_proc = subprocess.Popen(
            tr1_cmd.split(" "),
            stdout=subprocess.PIPE,
            env={"CUDA_VISIBLE_DEVICES": "2,3"})
        print("trainer proc started")

        tr0_out = tr0_proc.stdout.read()
        # check tr0_out

        ps0.terminate()
        ps1.terminate()

        # os.kill(ps0.pid, signal.SIGTERM)
        # os.kill(ps1.pid, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()

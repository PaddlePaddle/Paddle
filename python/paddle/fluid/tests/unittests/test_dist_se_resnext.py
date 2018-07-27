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
        self._python_interp = "python"

    def start_pserver(self):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps0_cmd = "%s dist_se_resnext.py pserver %s 0 %s %d TRUE" % \
            (self._python_interp, self._ps_endpoints, ps0_ep, self._trainers)
        ps1_cmd = "%s dist_se_resnext.py pserver %s 0 %s %d TRUE" % \
            (self._python_interp, self._ps_endpoints, ps1_ep, self._trainers)

        ps0_proc = subprocess.Popen(
            ps0_cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ps1_proc = subprocess.Popen(
            ps1_cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ps0_proc, ps1_proc

    def _wait_ps_ready(self, pid):
        retry_times = 20
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def test_with_place(self):
        # *ATTENTION* THIS TEST NEEDS AT LEAST 2GPUS TO RUN
        required_envs = {
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15"
        }
        # Run local to get a base line
        env_local = {"CUDA_VISIBLE_DEVICES": "0"}
        env_local.update(required_envs)
        local_cmd = "%s dist_se_resnext.py trainer %s 0 %s %d FLASE" % \
            (self._python_interp, "127.0.0.1:1234", "127.0.0.1:1234", 1)
        local_proc = subprocess.Popen(
            local_cmd.split(" "), stdout=subprocess.PIPE, env=env_local)
        local_proc.wait()
        local_ret = local_proc.stdout.read()

        # Run dist train to compare with local results
        ps0, ps1 = self.start_pserver()
        self._wait_ps_ready(ps0.pid)
        self._wait_ps_ready(ps1.pid)

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        tr0_cmd = "%s dist_se_resnext.py trainer %s 0 %s %d TRUE" % \
            (self._python_interp, self._ps_endpoints, ps0_ep, self._trainers)
        tr1_cmd = "%s dist_se_resnext.py trainer %s 1 %s %d TRUE" % \
            (self._python_interp, self._ps_endpoints, ps1_ep, self._trainers)

        env0 = {"CUDA_VISIBLE_DEVICES": "0"}
        env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        env0.update(required_envs)
        env1.update(required_envs)
        FNULL = open(os.devnull, 'w')

        tr0_proc = subprocess.Popen(
            tr0_cmd.split(" "), stdout=subprocess.PIPE, stderr=FNULL, env=env0)
        tr1_proc = subprocess.Popen(
            tr1_cmd.split(" "), stdout=subprocess.PIPE, stderr=FNULL, env=env1)

        tr0_proc.wait()
        tr1_proc.wait()
        loss_data0 = tr0_proc.stdout.read()
        lines = loss_data0.split("\n")
        dist_first_loss = eval(lines[0].replace(" ", ","))[0]
        dist_last_loss = eval(lines[1].replace(" ", ","))[0]

        local_lines = local_ret.split("\n")
        local_first_loss = eval(local_lines[0])[0]
        local_last_loss = eval(local_lines[1])[0]

        self.assertAlmostEqual(local_first_loss, dist_first_loss)
        self.assertAlmostEqual(local_last_loss, dist_last_loss)

        # check tr0_out
        # FIXME: ensure the server process is killed
        # replace with ps0.terminate()
        os.kill(ps0.pid, signal.SIGKILL)
        os.kill(ps1.pid, signal.SIGKILL)
        FNULL.close()


if __name__ == "__main__":
    unittest.main()

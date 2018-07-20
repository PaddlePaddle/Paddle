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

import unittest
import subprocess
import time
import argparse
import sys
import os
import signal
from contextlib import closing
import socket

JOB_ROLE_MASTER = "MASTER"
JOB_ROLE_TRAINER = "TRAINER"
JOB_ROLE_PSERVERS = "PSERVER"

# detect cuda device count automatically
CUDA_DEVICE_CNT = 2


class FluidDistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = ""
        cls.place = None

    def _wait_ps_ready(self, pid):
        retry_times = 5
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(1)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def _find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    def _ps_health(self, p):
        if p is None and p != 0:
            return False
        else:
            return True

    def launch_job(self, trainers, pservers):
        """
        This function will launch a distributed training job and hung until
        one of trainers or pservers exit.
        """
        role = os.getenv("PADDLE_TRAINING_ROLE", "MASTER")
        ps_endpoints = ",".join([
            "127.0.0.1:%d" % self._find_free_port() for _ in xrange(pservers)
        ])

        if role == JOB_ROLE_MASTER:
            p_list = []

            for i in xrange(pservers):
                p_env = os.environ
                p_env["PADDLE_TRAINING_ROLE"] = "PSERVER"
                p_env["PADDLE_PSERVER_ENDPOINTS"] = ps_endpoints
                p_env["PADDLE_CURRENT_ENDPOINT"] = ps_endpoints.split(",")[i]
                p_env["PADDLE_TRAINERS"] = str(trainers)
                p = subprocess.Popen(self.entry.split(), env=p_env)
                p_list.append(p)
                self._wait_ps_ready(p.pid)

            for i in xrange(trainers):
                p_env = os.environ
                p_env["PADDLE_TRAINING_ROLE"] = "TRAINER"
                p_env["PADDLE_TRAINER_ID"] = str(i)
                p_env["PADDLE_TRAINERS"] = str(trainers)
                p_env["CUDA_VISIBLE_DEVICES"] = str(trainers % CUDA_DEVICE_CNT)
                p_env["FLAGS_fraction_of_gpu_memory_to_use"] = "0.15"
                p_list.append(subprocess.Popen(self.entry.split(), env=p_env))

            job_failed = False
            exit_flag = False
            while not exit_flag:
                # get all the sub process' retcode
                polls = [p.poll() for p in p_list if p.poll() is not None]
                if len(polls) == 0:
                    # all processes are running
                    time.sleep(1)
                else:
                    if all(v == 0 for v in polls):
                        exit_flag = True
                    else:
                        exit_flag = True
                        job_failed = True

            # kill all the sub process
            [os.kill(p.pid, signal.SIGTERM) for p in p_list if p.poll() == None]

            if job_failed:
                raise AssertionError(
                    "sub process error, distributed training job failed.")

            [p.wait() for p in p_list]

        elif role == JOB_ROLE_PSERVERS:
            self.start_pserver()

        elif role == JOB_ROLE_TRAINER:
            self.start_trainer()

    def start_trainer(self):
        raise RuntimeError("should implement start_trainer interface.")

    def start_pserver(self):
        raise RuntimeError("should implement start_pserver interface.")

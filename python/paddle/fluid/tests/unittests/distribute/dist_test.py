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

JOB_ROLE_MASTER = "MASTER"
JOB_ROLE_TRAINER = "TRAINER"
JOB_ROLE_PSERVERS = "PSERVER"


class FluidDist(object):
    def __init__(self, entry, program=None):
        self.entry = entry
        self.prog = program

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
            s.bind(('', 0))
            return s.getsockname()[1]

    def _ps_health(self, p):
        if p is None and p != 0:
            return False
        else:
            return True

    def _get_env(self, role, trainers, pservers):
        env = os.environ
        ps_endpoints = ",".join([
            "127.0.0.1:%d" % self._find_free_port() for _ in xrange(pservers)
        ])
        env["PADDLE_TRAINING_ROLE"] = role

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

            for _ in xrange(pservers):
                p_env = os.environ
                p_env["PADDLE_TRAINING_ROLE"] = "PSERVER"
                p_env["PADDLE_PSERVER_ENDPOINTS"] = ps_endpoints
                p = subprocess.Popen(self.entry.split(), env=p_env)
                p_list.append(p)
                #self._wait_ps_ready(p.pid)

            for _ in xrange(trainers):
                p_env = os.environ
                p_env["PADDLE_TRAINING_ROLE"] = "TRAINER"
                p_list.append(subprocess.Popen(self.entry.split(), env=p_env))

            job_failed = False
            exit_flag = False
            while not exit_flag:
                # get all the sub process' retcode
                polls = [p.poll() for p in p_list if p.poll() is not None]
                if len(polls) == 0:
                    # all process is running
                    time.sleep(1)
                else:
                    if all(v == 0 for v in polls):
                        exit_flag = True
                    else:
                        exit_flag = True
                        job_failed = True

            # kill all the sub process
            [os.kill(p.pid, signal.SIGTERM) for p in p_list]

            if job_failed:
                raise AssertionError(
                    "sub process error, distributed training job failed.")

        elif role == JOB_ROLE_PSERVERS:
            self.start_pserver()

        elif role == JOB_ROLE_TRAINER:
            self.start_trainer()

    def start_trainer(self):
        raise RuntimeError("should implement start_trainer interface.")

    def start_pserver(self):
        raise RuntimeError("should implement start_pserver interface.")


if __name__ == "__main__":
    fd = FluidDist(entry="python dist_test.py")
    fd.launch_job(2, 2)

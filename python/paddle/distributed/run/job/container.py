# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from paddle.distributed.run.utils.process_context import ProcessContext

import os, copy, sys
import time
'''
A container can be run by process or just a callable function
'''


class ContainerStatus:
    UNINIT = "uninit"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATING = "terminating"
    UNKNOWN = "unknown"
    COMPLETED = "completed"


class Container(object):
    def __init__(self):
        self.entrypoint = []
        self.rank = 0
        self.retry: int = 3
        self.stdin = None
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.env = {}
        self.proc = None
        self.grace_period = 10

    def update_env(self, env={}, **kwargs):
        self.env.update(env)
        self.env.update(kwargs)

    def start(self, timeout=-1):
        st = time.time()

        if self.proc and self.proc.alive():
            return True

        self.proc = ProcessContext(
            self.entrypoint, env=self.env, out=self.stdout, err=self.stderr)
        self.proc.start()

        while timeout > 0 and time.time() - st < timeout:
            if self.proc.alive():
                time.sleep(0.1)
                continue
            if self.proc.exit_code() == 0:
                return True
            return False

    def terminate(self, force=False):
        if self.proc and self.proc.alive():
            return self.proc.terminate(force)

    def wait(self, timeout=None):
        self.proc.wait(timeout)

    def status(self):
        if not self.proc:
            return ContainerStatus.UNINIT
        if self.proc.alive():
            return ContainerStatus.RUNNING
        elif self.proc.exit_code() == 0:
            return ContainerStatus.COMPLETED
        else:
            return ContainerStatus.UNKNOWN

    def __str__(self):
        return 'Container {} {} {}'.format(self.env, self.entrypoint, self.rank)

    def logs(self):
        pass

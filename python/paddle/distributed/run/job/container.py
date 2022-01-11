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
        self.env = os.environ
        self.proc = None

    def update_env(self, env={}, **kwargs):
        self.env.update(env)
        self.env.update(kwargs)

    def run(self):
        self.proc = ProcessContext(
            self.entrypoint, env=self.env, out=self.stdout, err=self.stderr)
        self.proc.start()

    def exit(self):
        if self.proc.alive():
            self.proc.stop()

    def status(self):
        if not self.proc:
            return ContainerStatus.UNINIT
        if self.proc.alive():
            return ContainerStatus.RUNNING
        elif self.proc.exit_code() == 0:
            return ContainerStatus.COMPLETED
        else:
            return ContainerStatus.UNKNOWN

    def logs(self):
        pass

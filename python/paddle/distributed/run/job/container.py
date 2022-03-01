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

from .status import Status

import os, copy, sys
import time
'''
A container can be run by process or just a callable function
'''


class Container(object):
    def __init__(self):
        self.entrypoint = []
        self.rank = 0
        self.retry: int = 3
        self.out = None
        self.err = None
        self.env = {}
        self.proc = None
        self.grace_period = 10

        self._log_handler = None

    def update_env(self, env={}, **kwargs):
        env = {k: v for k, v in env.items() if isinstance(v, str)}
        self.env.update(env)

        kwargs = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self.env.update(kwargs)

    def _get_fd(self, pth):
        if not pth:
            return None

        try:
            d = os.path.dirname(pth)
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            return open(pth, 'w')
        except:
            return None

    def start(self, timeout=-1):
        st = time.time()

        if self.proc and self.proc.alive():
            return True

        self.stdout = self._get_fd(self.out) or sys.stdout
        if self.out == self.err:
            self.stderr = self.stdout
        elif self.err:
            self.stderr = self._get_fd(self.err) or sys.stderr

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
        if self._log_handler:
            self._log_handler.close()
            self._log_handler = None

        if self.proc and self.proc.alive():
            return self.proc.terminate(force)

    def wait(self, timeout=None):
        self.proc.wait(timeout)

    def exit_code(self):
        return self.proc.exit_code()

    def status(self):
        if not self.proc:
            return Status.UNINIT
        if self.proc.alive():
            return Status.RUNNING
        elif self.proc.exit_code() == 0:
            return Status.COMPLETED
        else:
            return Status.FAILED

    def __str__(self):
        return 'Container env {} cmd {} rank {}'.format(
            self.env, self.entrypoint, self.rank)

    def logs(self, fn=None, offset=-1):
        if not self._log_handler:
            self._log_handler = open(self.out)

        if offset >= 0:
            self._log_handler.seek(offset, 0)

        if fn is None:
            fn = sys.stdout

        try:
            for line in self._log_handler:
                fn.write(line)
        finally:
            return self._log_handler.tell()

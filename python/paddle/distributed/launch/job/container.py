# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

from paddle.distributed.launch.utils.process_context import ProcessContext

from .status import Status


class Container:
    '''
    TODO(kuizhiqing) A container can be run by process/thread or just a callable function
    '''

    def __init__(self, entrypoint=[], rank=-1, env={}, overwrite_log=False):
        self._entrypoint = entrypoint
        self._rank = rank
        self._out = None
        self._err = None
        self._env = env
        self._proc = None

        self._retry: int = 3
        self._grace_period = 10

        self._log_handler = None
        self._shell = False

        self.log_mode = 'w' if overwrite_log else 'a'

    @property
    def env(self):
        return self._env

    @property
    def entrypoint(self):
        return self._entrypoint

    @entrypoint.setter
    def entrypoint(self, entry):
        self._entrypoint = entry

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, r):
        self._rank = r

    @property
    def outfile(self):
        return self._out

    @outfile.setter
    def outfile(self, out):
        self._out = out

    @property
    def errfile(self):
        return self._err

    @errfile.setter
    def errfile(self, err):
        self._err = err

    @property
    def shell(self):
        return self._shell

    @shell.setter
    def shell(self, shell):
        self._shell = shell

    def update_env(self, env={}, **kwargs):
        env = {k: v for k, v in env.items() if isinstance(v, str)}
        self._env.update(env)

        kwargs = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self._env.update(kwargs)

    def _validate_env(self):
        for k, v in self._env.items():
            assert isinstance(k, str) and isinstance(
                v, str
            ), f'env {k}:{v} must be str'

    def _get_fd(self, pth):
        if not pth:
            return None

        try:
            d = os.path.dirname(pth)
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            return open(pth, self.log_mode)
        except:
            return None

    def start(self):
        if self._proc and self._proc.alive():
            return True

        self._validate_env()

        self._stdout = self._get_fd(self._out) or sys.stdout
        if self._out == self._err:
            self._stderr = self._stdout
        elif self._err:
            self._stderr = self._get_fd(self._err) or sys.stderr

        if self._out and not self._log_handler:
            self._log_handler = open(self._out)
            self._log_handler.seek(0, 2)
            self._log_start_offset = self._log_handler.tell()

        self._proc = ProcessContext(
            self._entrypoint,
            env=self._env,
            out=self._stdout,
            err=self._stderr,
            shell=self._shell,
        )

        self._proc.start()

    def terminate(self, force=False):
        if self._log_handler:
            self._log_handler.close()
            self._log_handler = None

        if self._proc and self._proc.alive():
            return self._proc.terminate(force)

    def wait(self, timeout=None):
        try:
            self._proc.wait(timeout)
            return True
        except Exception:
            return False

    @property
    def exit_code(self):
        return self._proc.exit_code() if self._proc else -1

    @property
    def status(self):
        if not self._proc:
            return Status.UNINIT
        if self._proc.alive():
            return Status.RUNNING
        elif self._proc.exit_code() == 0:
            return Status.COMPLETED
        else:
            return Status.FAILED

    def __str__(self):

        need_print = os.environ.get('FLAGS_print_launcher_env', 'false').lower()
        if need_print == 'true' or need_print == '1':
            return f'Container rank {self._rank} status {self.status} cmd {self._entrypoint} code {self.exit_code} log {self.errfile} \nenv {self._env}'
        return f'Container rank {self._rank} status {self.status} cmd {self._entrypoint} code {self.exit_code} log {self.errfile}'

    def logs(self, fn=None, offset=0, whence=1, limit=1000):
        if not self._log_handler:
            return

        if fn is None:
            fn = sys.stdout

        try:
            if offset != 0 or whence != 1:
                if whence == 0 and offset < self._log_start_offset:
                    offset = self._log_start_offset
                self._log_handler.seek(offset, whence)

            for _ in range(limit):
                line = self._log_handler.readline()
                if not line:
                    return False
                fn.write(line)
            return True
        except:
            return

    def tail(self, length=3000):
        if not self._log_handler:
            return

        try:
            self._log_handler.seek(0, 2)
            ed = self._log_handler.tell()
        except:
            pass

        if ed > length:
            self.logs(offset=ed - length, whence=0)
        else:
            self.logs(offset=0, whence=0)

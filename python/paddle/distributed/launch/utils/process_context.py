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

import subprocess
import os, sys, signal, time


class ProcessContext(object):
    def __init__(self,
                 cmd,
                 env=os.environ,
                 out=sys.stdout,
                 err=sys.stderr,
                 group=True,
                 preexec_fn=None):
        self._cmd = cmd
        self._env = env
        self._preexec_fn = preexec_fn
        self._stdout = out
        self._stderr = err
        self._group = group if os.name != 'nt' else False
        self._proc = None
        self._code = None

    def _start(self):
        pre_fn = os.setsid if self._group else None
        self._proc = subprocess.Popen(
            self._cmd,
            env=self._env,
            stdout=self._stdout,
            stderr=self._stderr,
            preexec_fn=self._preexec_fn or pre_fn)

    def _close_std(self):
        try:
            if not self._stdout.isatty():
                self._stdout.close()

            if not self._stderr.isatty():
                self._stderr.close()
        except:
            pass

    def alive(self):
        return self._proc and self._proc.poll() is None

    def exit_code(self):
        return self._proc.poll() if self._proc else None

    def start(self):
        self._start()

    def terminate(self, force=False, max_retry=3):
        for i in range(max_retry):
            if self.alive():
                if self._group:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:
                    self._proc.terminate()
                time.sleep(0.2)
            else:
                break

        if force and self.alive():
            self._proc.kill()

        self._close_std()

        return self.alive()

    def wait(self, timeout=None):
        self._proc.wait(timeout)

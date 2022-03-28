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


class Status(object):
    UNINIT = "uninit"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATING = "terminating"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    DONE = "done"  # should exit whatever status

    def __init__(self):
        self._current_status = None

    def current(self):
        return self._current_status

    def is_running(self):
        return self._current_status == self.RUNNING

    def is_restarting(self):
        return self._current_status == self.RESTARTING

    def is_done(self):
        if self._current_status in [self.DONE, self.COMPLETED, self.FAILED]:
            return True
        else:
            return False

    def run(self):
        self._current_status = self.RUNNING

    def fail(self):
        self._current_status = self.FAILED

    def complete(self):
        self._current_status = self.COMPLETED

    def restart(self):
        self._current_status = self.RESTARTING

    def done(self):
        self._current_status = self.DONE

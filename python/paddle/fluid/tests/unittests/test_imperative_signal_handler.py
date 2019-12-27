# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import signal
import unittest
import multiprocessing
import time

from paddle.fluid import core


def set_child_signal_handler(self, child_pid):
    core._set_process_pid(id(self), child_pid)
    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)

    signal.signal(signal.SIGCHLD, __handler__)


class TestDygraphDataLoaderSingalHandler(unittest.TestCase):
    def kill_child_process_by_signal(self, sig):
        def __test_process__():
            core._set_process_signal_handler()
            time.sleep(1)
            os.kill(os.getpid(), sig)

        test_process = multiprocessing.Process(target=__test_process__)
        test_process.daemon = True
        test_process.start()

        set_child_signal_handler(id(self), test_process.pid)
        test_process.join()

    def test_child_process_killed_by_sigsegv(self):
        self.kill_child_process_by_signal(signal.SIGSEGV)

    def test_child_process_killed_by_sigbus(self):
        self.kill_child_process_by_signal(signal.SIGBUS)

    def test_child_process_killed_by_sigterm(self):
        self.kill_child_process_by_signal(signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()

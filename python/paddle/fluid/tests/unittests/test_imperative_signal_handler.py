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

import paddle.compat as cpt
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard


def set_child_signal_handler(self, child_pid):
    core._set_process_pids(id(self), tuple([child_pid]))
    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)

    signal.signal(signal.SIGCHLD, __handler__)


class DygraphDataLoaderSingalHandler(unittest.TestCase):
    def func_child_process_exit_with_error(self):
        def __test_process__():
            core._set_process_signal_handler()
            sys.exit(1)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Fatal", cpt.get_exception_message(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_exit_with_error(self):
        with _test_eager_guard():
            self.func_child_process_exit_with_error()
        self.func_child_process_exit_with_error()

    def func_child_process_killed_by_sigsegv(self):
        def __test_process__():
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGSEGV)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Segmentation fault",
                              cpt.get_exception_message(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigsegv(self):
        with _test_eager_guard():
            self.func_child_process_killed_by_sigsegv()
        self.func_child_process_killed_by_sigsegv()

    def func_child_process_killed_by_sigbus(self):
        def __test_process__():
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGBUS)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Bus error", cpt.get_exception_message(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigbus(self):
        with _test_eager_guard():
            self.func_child_process_killed_by_sigbus()
        self.func_child_process_killed_by_sigbus()

    def func_child_process_killed_by_sigterm(self):
        def __test_process__():
            core._set_process_signal_handler()
            time.sleep(10)

        test_process = multiprocessing.Process(target=__test_process__)
        test_process.daemon = True
        test_process.start()

        set_child_signal_handler(id(self), test_process.pid)
        time.sleep(1)

    def test_child_process_killed_by_sigterm(self):
        with _test_eager_guard():
            self.func_child_process_killed_by_sigterm()
        self.func_child_process_killed_by_sigterm()


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import multiprocessing
import queue
import signal
import time
import unittest

from paddle.fluid.reader import (
    CleanupFuncRegistrar,
    _cleanup,
    multiprocess_queue_set,
)
=======
import sys
import signal
import unittest
import multiprocessing
import time

import paddle.compat as cpt
from paddle.fluid.framework import _test_eager_guard

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

from paddle.fluid.reader import multiprocess_queue_set, _cleanup, CleanupFuncRegistrar
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

# NOTE: These special functions cannot be detected by the existing coverage mechanism,
# so the following unittests are added for these internal functions.


class TestDygraphDataLoaderCleanUpFunc(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.capacity = 10

    def test_clear_queue_set(self):
        test_queue = queue.Queue(self.capacity)
=======

    def setUp(self):
        self.capacity = 10

    def func_test_clear_queue_set(self):
        test_queue = queue.Queue(self.capacity)
        global multiprocess_queue_set
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        multiprocess_queue_set.add(test_queue)
        for i in range(0, self.capacity):
            test_queue.put(i)
        _cleanup()

<<<<<<< HEAD
=======
    def test_clear_queue_set(self):
        with _test_eager_guard():
            self.func_test_clear_queue_set()
        self.func_test_clear_queue_set()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

class TestRegisterExitFunc(unittest.TestCase):
    # This function does not need to be implemented in this case
    def none_func(self):
        pass

<<<<<<< HEAD
    def test_not_callable_func(self):
=======
    def func_test_not_callable_func(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        exception = None
        try:
            CleanupFuncRegistrar.register(5)
        except TypeError as ex:
<<<<<<< HEAD
            self.assertIn("is not callable", str(ex))
            exception = ex
        self.assertIsNotNone(exception)

    def test_old_handler_for_sigint(self):
        CleanupFuncRegistrar.register(
            function=self.none_func, signals=[signal.SIGINT]
        )

    def test_signal_wrapper_by_sigchld(self):
=======
            self.assertIn("is not callable", cpt.get_exception_message(ex))
            exception = ex
        self.assertIsNotNone(exception)

    def test_not_callable_func(self):
        with _test_eager_guard():
            self.func_test_not_callable_func()
        self.func_test_not_callable_func()

    def func_test_old_handler_for_sigint(self):
        CleanupFuncRegistrar.register(function=self.none_func,
                                      signals=[signal.SIGINT])

    def test_old_handler_for_sigint(self):
        with _test_eager_guard():
            self.func_test_old_handler_for_sigint()
        self.func_test_old_handler_for_sigint()

    def func_test_signal_wrapper_by_sigchld(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # This function does not need to be implemented in this case
        def __test_process__():
            pass

<<<<<<< HEAD
        CleanupFuncRegistrar.register(
            function=self.none_func, signals=[signal.SIGCHLD]
        )
=======
        CleanupFuncRegistrar.register(function=self.none_func,
                                      signals=[signal.SIGCHLD])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        exception = None
        try:
            test_process = multiprocessing.Process(target=__test_process__)
            test_process.start()
            time.sleep(3)
        except SystemExit as ex:
            exception = ex
        self.assertIsNotNone(exception)

<<<<<<< HEAD
=======
    def test_signal_wrapper_by_sigchld(self):
        with _test_eager_guard():
            self.func_test_signal_wrapper_by_sigchld()
        self.func_test_signal_wrapper_by_sigchld()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()

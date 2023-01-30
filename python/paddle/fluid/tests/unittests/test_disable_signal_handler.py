#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import signal
import subprocess
import unittest

SignalsToTest = {
    signal.SIGTERM,
    signal.SIGBUS,
    signal.SIGABRT,
    signal.SIGSEGV,
    signal.SIGILL,
    signal.SIGFPE,
=======
from __future__ import print_function

import unittest
import numpy as np
import signal, os
import paddle
import subprocess

SignalsToTest = {
    signal.SIGTERM, signal.SIGBUS, signal.SIGABRT, signal.SIGSEGV,
    signal.SIGILL, signal.SIGFPE
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}


class TestSignOpError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        try:
            for sig in SignalsToTest:
                output = subprocess.check_output(
                    [
                        "python",
                        "-c",
                        f"import paddle; import signal,os; paddle.disable_signal_handler(); os.kill(os.getpid(), {sig})",
                    ],
                    stderr=subprocess.STDOUT,
                )
=======

    def test_errors(self):
        try:
            for sig in SignalsToTest:
                output = subprocess.check_output([
                    "python", "-c",
                    f"import paddle; import signal,os; paddle.disable_signal_handler(); os.kill(os.getpid(), {sig})"
                ],
                                                 stderr=subprocess.STDOUT)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        except Exception as e:
            # If paddle signal handler is enabled
            # One would expect "paddle::framework::SignalHandle" in STDERR
            stdout_message = str(e.output)
            if "paddle::framework::SignalHandle" in stdout_message:
                raise Exception("Paddle signal handler not disabled")


if __name__ == "__main__":
    unittest.main()

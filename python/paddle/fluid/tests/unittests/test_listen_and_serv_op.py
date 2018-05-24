#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import signal
import time
import thread
import unittest
import numpy as np
from op_test import OpTest


class TestListenAndServOp(OpTest):
    def setUp(self):
        self.sleep_time = 2
        self.op_type = "listen_and_serv"
        self.inputs = {'X': {[]}}
        self.attrs = {
            'endpoint': '127.0.0.1:6173',
            'sync_mode': True
        }
        self.output = {
        }

    def _raise_signal(sleep_time, signal_to_be_handled)
        time.sleep(self.sleep_time)
        os.kill(os.getpid(), signal_to_be_handled)

    def _async_raise_signal(self, signal_to_be_handled):
        print "wait for signal: %s, in %s second" % (signal_to_be_handled, self.sleep_time)
        thread.start_new_thread(_raise_signal, (self.sleep_time, signal_to_be_handled))

    def test_handle_sigint_in_sync_mode(self):
        print "start test_handle_sigint_in_sync_mode"
        self._async_raise_signal(signal.SIGINT)
        self.check_output()

    def test_handle_sigterm_in_sync_mode(self):
        print "start test_handle_sigint_in_sync_mode"
        self._async_raise_signal(signal.SIGTERM)
        self.check_output()

    def test_handle_sigint_in_sync_mode(self):
        print "start test_handle_sigint_in_sync_mode"
        self.attrs = {
            'endpoint': '127.0.0.1:6173',
            'sync_mode': False
        }
        self._async_raise_signal(signal.SIGTERM)
        self.check_output()

    def test_handle_sigint_in_sync_mode(self):
        print "start test_handle_sigint_in_sync_mode"
        self.attrs = {
            'endpoint': '127.0.0.1:6173',
            'sync_mode': False
        }
        self._async_raise_signal(signal.SIGTERM)
        self.check_output()


if __name__ == '__main__':
    unittest.main()

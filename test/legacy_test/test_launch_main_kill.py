# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import platform
import signal
import unittest
from unittest import mock

from paddle.distributed.utils import launch_utils


class TestLaunchMainProcessCleanup(unittest.TestCase):
    def test_filter_pids(self):
        processes = [" 123 ", "456", "abc", "", "789"]
        self_pid = 456

        result = launch_utils.filter_pids(processes, self_pid)

        self.assertEqual(result, [123, 789])

    @mock.patch("paddle.distributed.utils.launch_utils.os.kill")
    def test_terminate_processes_process_lookup(self, mock_kill):
        pids = [12345, 67890]
        mock_kill.side_effect = [None, ProcessLookupError()]
        expected_sig = (
            signal.SIGKILL if platform.system() != "Windows" else signal.SIGTERM
        )

        result = launch_utils.terminate_processes(pids)

        self.assertTrue(result)
        self.assertEqual(mock_kill.call_count, 2)
        mock_kill.assert_has_calls(
            [
                mock.call(12345, expected_sig),
                mock.call(67890, expected_sig),
            ],
            any_order=False,
        )

    @mock.patch("paddle.distributed.utils.launch_utils.os.kill")
    def test_terminate_processes_permission_error(self, mock_kill):
        pids = [12345, 67890]
        mock_kill.side_effect = [ProcessLookupError(), PermissionError()]
        expected_sig = (
            signal.SIGKILL if platform.system() != "Windows" else signal.SIGTERM
        )

        result = launch_utils.terminate_processes(pids)

        self.assertFalse(result)
        self.assertEqual(mock_kill.call_count, 2)
        mock_kill.assert_has_calls(
            [
                mock.call(12345, expected_sig),
                mock.call(67890, expected_sig),
            ],
            any_order=False,
        )


if __name__ == "__main__":
    unittest.main()

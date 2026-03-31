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

import unittest
from unittest import mock

from paddle.distributed.utils import launch_utils


class TestLaunchMainProcessCleanup(unittest.TestCase):
    @mock.patch("paddle.distributed.utils.launch_utils.os.kill")
    def test_terminate_other_processes_swallow_errors(self, mock_kill):
        processes = ["", "abc", "12345", " 67890 ", "99999"]
        self_pid = "99999"
        expected_calls = [mock.call(12345, 9), mock.call(67890, 9)]

        mock_kill.side_effect = ProcessLookupError()
        try:
            launch_utils.terminate_other_processes(processes, self_pid)
        except ProcessLookupError:
            self.fail(
                "terminate_other_processes should swallow ProcessLookupError"
            )
        self.assertEqual(mock_kill.call_count, 2)
        mock_kill.assert_has_calls(expected_calls, any_order=False)

        mock_kill.reset_mock()
        mock_kill.side_effect = PermissionError()
        try:
            launch_utils.terminate_other_processes(processes, self_pid)
        except PermissionError:
            self.fail("terminate_other_processes should swallow PermissionError")
        self.assertEqual(mock_kill.call_count, 2)
        mock_kill.assert_has_calls(expected_calls, any_order=False)


if __name__ == "__main__":
    unittest.main()

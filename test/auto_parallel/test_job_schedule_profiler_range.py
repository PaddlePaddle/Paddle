# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.profiler.utils import job_schedule_profiler_range


class TestJobScheDuleProfilerRange(unittest.TestCase):
    def test_not_exit_after_prof_1(self):
        status_list = [
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        for i in range(10):
            with job_schedule_profiler_range(i, 3, 6, False) as status:
                self.assertEqual(status, status_list[i])

    def test_not_exit_after_prof_2(self):
        status_list = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        for i in range(10):
            with job_schedule_profiler_range(i, 0, 5, False) as status:
                self.assertEqual(status, status_list[i])

    def test_not_exit_after_prof_3(self):
        status_list = [
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        for i in range(10):
            with job_schedule_profiler_range(i, 3, 5, False) as status:
                self.assertEqual(status, status_list[i])

    def test_end_less_than_start(self):
        status_list = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        for i in range(10):
            with job_schedule_profiler_range(i, 5, 3, False) as status:
                self.assertEqual(status, status_list[i])


if __name__ == "__main__":
    unittest.main()

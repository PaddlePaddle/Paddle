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

import os
import subprocess
import sys
import tempfile
import unittest


class TestVPPPass(unittest.TestCase):
    def test_pp2(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(
            file_dir, "pipeline_scheduler_vpp_unittest_deprecated.py"
        )

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        tmp_dir = tempfile.TemporaryDirectory()
        cmd = [
            sys.executable,
            "-u",
            *coverage_args,
            "-m",
            "paddle.distributed.launch",
            "--devices",
            "0,1",
            "--log_dir",
            tmp_dir.name,
            launch_model_path,
        ]

        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

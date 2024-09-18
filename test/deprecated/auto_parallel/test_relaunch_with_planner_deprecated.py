# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.append("../../auto_parallel")


class TestPlannerReLaunch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_relaunch_with_planner(self):
        from test_auto_parallel_relaunch_deprecated import (
            cluster_json,
            mapping_json,
        )

        cluster_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_cluster.json"
        )
        mapping_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_rank_mapping.json"
        )

        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

        mapping_json_object = json.loads(mapping_json)
        with open(mapping_json_path, "w") as mapping_json_file:
            json.dump(mapping_json_object, mapping_json_file)

        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(
            file_dir, "auto_parallel_relaunch_with_planner_deprecated.py"
        )

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        cmd = [
            sys.executable,
            "-u",
            *coverage_args,
            "-m",
            "paddle.distributed.launch",
            "--log_dir",
            self.temp_dir.name,
            "--cluster_topo_path",
            cluster_json_path,
            "--rank_mapping_path",
            mapping_json_path,
            "--enable_auto_mapping",
            "True",
            launch_model_path,
        ]
        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()

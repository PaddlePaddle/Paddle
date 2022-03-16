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

import unittest
import os
import sys
import json
import shutil
import subprocess
from paddle.distributed.fleet.launch_utils import run_with_coverage


class TestPlannerReLaunch(unittest.TestCase):
    def test_relaunch_with_planner(self):
        from test_auto_parallel_relaunch import cluster_json
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

        launch_model_path = os.path.join(
            file_dir, "auto_parallel_relaunch_with_gpt_planner.py")

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        cmd = [sys.executable, "-u"] + coverage_args + [
            "-m", "launch", "--cluster_topo_path", cluster_json_path,
            "--enable_auto_mapping", "True", launch_model_path
        ]
        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)
        rank_mapping_json_path = os.path.join(file_dir,
                                              "auto_parallel_rank_mapping.json")
        if os.path.exists(rank_mapping_json_path):
            os.remove(rank_mapping_json_path)
        log_path = os.path.join(file_dir, "log")
        if os.path.exists(log_path):
            shutil.rmtree(log_path)


if __name__ == "__main__":
    unittest.main()

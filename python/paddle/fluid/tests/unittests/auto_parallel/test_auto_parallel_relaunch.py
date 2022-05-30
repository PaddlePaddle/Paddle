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

cluster_json = """
{
  "machines": [
    {
      "hostname": "machine1",
      "addr": "127.0.0.1",
      "port": "768",
      "devices": [
        {
          "global_id": 0,
          "local_id": 0,
          "type": "GPU",
          "model": "Tesla V100-SXM2-32GB",
          "sp_gflops": 15700,
          "dp_gflops": 7800,
          "memory": 32
        },
        {
          "global_id": 1,
          "local_id": 1,
          "type": "GPU",
          "model": "Tesla V100-SXM2-32GB",
          "sp_gflops": 15700,
          "dp_gflops": 7800,
          "memory": 32
        },
        {
          "global_id": 2,
          "local_id": 0,
          "type": "CPU",
          "model": "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G",
          "arch": "x86_64",
          "vendor": "GenuineIntel",
          "sp_gflops": 150,
          "dp_gflops": 75,
          "memory": "503"
        }
      ],
      "links": [
        {
          "source_global_id": 0,
          "target_global_id": 1,
          "type": "NVL",
          "bandwidth": 42
        },
        {
          "source_global_id": 1,
          "target_global_id": 0,
          "type": "PHB",
          "bandwidth": 12
        }
      ]
    }
  ]
}
"""


class TestAutoParallelReLaunch(unittest.TestCase):
    def test_relaunch(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

        launch_model_path = os.path.join(file_dir,
                                         "auto_parallel_relaunch_model.py")

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

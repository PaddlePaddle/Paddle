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

<<<<<<< HEAD
import json
import os
import subprocess
import sys
import tempfile
import unittest
=======
import tempfile
import unittest
import os
import sys
import json
import shutil
import subprocess
from paddle.distributed.fleet.launch_utils import run_with_coverage
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

mapping_josn = """
[
  {
<<<<<<< HEAD
    "hostname": "machine1",
    "addr": "127.0.0.1",
    "port": "768",
    "ranks":
      {
        "0": [1],
=======
    "hostname": "machine1", 
    "addr": "127.0.0.1", 
    "port": "768", 
    "ranks": 
      {
        "0": [1], 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        "1": [0]
      }
  }
]
"""


class TestAutoParallelReLaunch(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_relaunch(self):
<<<<<<< HEAD
        cluster_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_cluster.json"
        )
        mapping_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_rank_mapping.json"
        )
=======
        cluster_json_path = os.path.join(self.temp_dir.name,
                                         "auto_parallel_cluster.json")
        mapping_json_path = os.path.join(self.temp_dir.name,
                                         "auto_parallel_rank_mapping.json")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

        mapping_josn_object = json.loads(mapping_josn)
        with open(mapping_json_path, "w") as mapping_josn_file:
            json.dump(mapping_josn_object, mapping_josn_file)

        file_dir = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
        launch_model_path = os.path.join(
            file_dir, "auto_parallel_relaunch_model.py"
        )
=======
        launch_model_path = os.path.join(file_dir,
                                         "auto_parallel_relaunch_model.py")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

<<<<<<< HEAD
        cmd = (
            [sys.executable, "-u"]
            + coverage_args
            + [
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
        )
=======
        cmd = [sys.executable, "-u"] + coverage_args + [
            "-m", "paddle.distributed.launch", "--log_dir", self.temp_dir.name,
            "--cluster_topo_path", cluster_json_path, "--rank_mapping_path",
            mapping_json_path, "--enable_auto_mapping", "True",
            launch_model_path
        ]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()

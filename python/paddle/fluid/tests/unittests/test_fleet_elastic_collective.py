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

from __future__ import print_function

import os
import time
import json
import unittest
import argparse
import tempfile
from warnings import catch_warnings

from paddle.distributed.fleet.elastic.collective import CollectiveLauncher

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


class TestCollectiveLauncher(unittest.TestCase):
    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))

        self.cluster_json_path = os.path.join(file_dir,
                                              "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(self.cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

    def test_launch(self):
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "1"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = "127.0.0.1"
            scale = None
            force = None
            backend = 'gloo'
            enable_auto_mapping = True
            run_mode = "cpuonly"
            servers = None
            cluster_topo_path = self.cluster_json_path
            rank_mapping_path = None
            training_script = "python -h"
            training_script_args = ["--use_amp false"]
            log_dir = None

        args = Argument()

        launch = CollectiveLauncher(args)
        try:
            launch.launch()
            launch.stop()

            args.rank_mapping_path = "./rank_mapping"
            launch.launch()
            launch.stop()
        except Exception as e:
            pass

        try:
            args.backend = "nccl"
            launch.launch()
            launch.stop()
        except Exception as e:
            pass

        try:
            args.backend = "unknown"
            args.enable_auto_mapping = True
            launch.launch()
            launch.stop()
        except Exception as e:
            pass

        try:
            paddle.distributed.fleet.launch.launch_collective(args)
        except Exception as e:
            pass

    def test_stop(self):
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "1"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = "127.0.0.1"
            scale = None
            force = None
            backend = 'gloo'
            enable_auto_mapping = True
            run_mode = "cpuonly"
            servers = None
            cluster_topo_path = self.cluster_json_path
            rank_mapping_path = None
            training_script = "python -h"
            training_script_args = ["--use_amp false"]
            log_dir = None

        args = Argument()
        try:
            launch = CollectiveLauncher(args)
            launch.tmp_dir = tempfile.mkdtemp()
            launch.stop()
        except Exception as e:
            pass


if __name__ == "__main__":
    unittest.main()

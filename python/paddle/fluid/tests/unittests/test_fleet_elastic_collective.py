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
import traceback
from warnings import catch_warnings

from paddle.distributed.fleet.elastic.collective import CollectiveLauncher
from paddle.distributed.fleet.launch import launch_collective

fake_python_code = """
print("test")
"""


class TestCollectiveLauncher(unittest.TestCase):
    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))

        self.code_path = os.path.join(file_dir, "fake_python_for_elastic.py")
        with open(self.code_path, "w") as f:
            f.write(fake_python_code)

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
            enable_auto_mapping = False
            run_mode = "cpuonly"
            servers = None
            rank_mapping_path = None
            training_script = "fake_python_for_elastic.py"
            training_script_args = ["--use_amp false"]
            log_dir = None

        args = Argument()

        launch = CollectiveLauncher(args)

        try:
            args.backend = "gloo"
            launch.launch()
            launch.stop()
        except Exception as e:
            pass

        try:
            args.backend = "gloo"
            launch_collective(args)
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
            enable_auto_mapping = False
            run_mode = "cpuonly"
            servers = None
            rank_mapping_path = None
            training_script = "fake_python_for_elastic.py"
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

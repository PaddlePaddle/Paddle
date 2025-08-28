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

import json
import os
import subprocess
import sys
import tempfile
import unittest


class TestEngineAPI(unittest.TestCase):
    def test_auto_tuner(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(
            file_dir, "engine_api_dp_deprecated.py"
        )

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []
        test_info = {
            "dp_degree": "auto",
            "mp_degree": "auto",
            "pp_degree": "auto",
            "micro_batch_size": "auto",
            "sharding_degree": "auto",
            "sharding_stage": "auto",
            "use_recompute": "auto",
            "recompute_granularity": "auto",
            "task_limit": 1,
            "max_time_per_task": 90,
            "model_cfg": {
                "hidden_size": 2048,
                "global_batch_size": 64,
                "num_layers": 24,
                "num_attention_heads": 16,
                "vocab_size": 50304,
            },
            "run_cmd": {
                "dp_degree": ["-o", "Distributed.dp_degree"],
                "mp_degree": ["-o", "Distributed.mp_degree"],
                "pp_degree": ["-o", "Distributed.pp_degree"],
                "micro_batch_size": ["-o", "Global.micro_batch_size"],
                "local_batch_size": ["-o", "Global.local_batch_size"],
                "sharding_degree": [
                    "-o",
                    "Distributed.sharding.sharding_degree",
                ],
                "sharding_stage": ["-o", "Distributed.sharding.sharding_stage"],
                "use_recompute": ["-o", "Model.use_recompute"],
                "recompute_granularity": ["-o", "Model.recompute_granularity"],
            },
            "metric_cfg": {
                "name": "ms/step",
                "OptimizationDirection": "Maximize",
            },
        }

        tmp_dir = tempfile.TemporaryDirectory()
        json_object = json.dumps(test_info)
        test_json_path = os.path.join(tmp_dir.name, "test.json")
        with open(test_json_path, "w") as f:
            f.write(json_object)

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
            "--auto_tuner_json",
            test_json_path,
            launch_model_path,
        ]

        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

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
#
# Launcher: runs hybrid_parallel_sharding_muon_no_hook_model.py on 8 GPUs to
# verify that the MuonShardingOptimizer "no-hook shared color" sync-only path
# under comm_overlap is bit-for-bit identical to the non-overlap baseline.

import time
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    get_cluster_from_args,
    get_devices,
    start_local_trainers,
)

from paddle import base
from paddle.distributed.utils.launch_utils import watch_local_trainers


class TestMuonNoHookOverlap(unittest.TestCase):
    def run_8accelerators(self, target_file_name, need_envs={}):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < 8
        ):
            return

        selected_devices = get_devices("0,1,2,3,4,5,6,7")
        cluster, pod = get_cluster_from_args(selected_devices)

        procs = start_local_trainers(
            cluster,
            pod,
            allocator_strategy="auto_growth",
            training_script=target_file_name,
            training_script_args=[],
            need_envs=need_envs,
            accelerator_type="gpu",
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())
            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)

    def test_muon_no_hook_overlap_matches_baseline(self):
        self.run_8accelerators(
            "hybrid_parallel_sharding_muon_no_hook_model.py",
        )


if __name__ == "__main__":
    unittest.main()

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

import time
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    get_cluster_from_args,
    get_devices,
    start_local_trainers,
    watch_local_trainers,
)

from paddle import base

WORKER = 'hybrid_parallel_pp_dw_recompute_overlap.py'


class TestPPDwRecomputeOverlap(unittest.TestCase):
    """4 accelerators, so pp_degree can be > 2 as the fillers need.

    ``TestMultipleAccelerators.run_mnist_2accelerators`` hardcodes devices 0,1;
    the dW / early-recompute windows only show their rank-divergent behaviour
    with more than two stages, so the launcher is spelled out here.
    """

    def _run_4accelerators(self, need_envs):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < 4
        ):
            return

        cluster, pod = get_cluster_from_args(get_devices('0,1,2,3'))
        procs = start_local_trainers(
            cluster,
            pod,
            allocator_strategy="auto_growth",
            training_script=WORKER,
            training_script_args=[],
            need_envs=need_envs,
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())
            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)

    def test_interleave(self):
        # accumulate_steps >= 2 * pp_degree -> PipelineParallelWithInterleave
        self._run_4accelerators(
            {"PP_DW_ACC_STEPS": "8", "PP_DW_BEST_UNBALANCED": "0"}
        )

    def test_vpp_fthenb_balanced_memory(self):
        # pp_degree <= accumulate_steps < 2 * pp_degree, with
        # best_unbalanced_scheduler -> VPPFhenBInBalancedMemory, the schedule
        # that opens three separate windows and reorders the last stage's
        # gradient targets.
        self._run_4accelerators(
            {"PP_DW_ACC_STEPS": "4", "PP_DW_BEST_UNBALANCED": "1"}
        )


if __name__ == "__main__":
    unittest.main()

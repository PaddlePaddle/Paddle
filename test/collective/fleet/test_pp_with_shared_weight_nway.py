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

"""4-accelerator launcher for the shared-weight N-way sync test.

The existing shared-weight tests all run pp_degree=2 (``run_mnist_2accelerators``),
which cannot cover the >=3-stage path this change fixes.  Spawn the worker
(``hybrid_parallel_pp_shared_weight_nway.py``) on 4 devices explicitly, skipping
only when fewer than 4 GPUs are available.
"""

import time
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    get_cluster_from_args,
    get_devices,
    start_local_trainers,
    watch_local_trainers,
)

from paddle import base

WORKER = 'hybrid_parallel_pp_shared_weight_nway.py'


class TestPPWithSharedWeightNWay(unittest.TestCase):
    """4 accelerators, so a shared key can live on 4 distinct PP stages."""

    def _run_4accelerators(self):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < 4
        ):
            self.skipTest("requires 4 CUDA devices")

        cluster, pod = get_cluster_from_args(get_devices('0,1,2,3'))
        procs = start_local_trainers(
            cluster,
            pod,
            allocator_strategy="auto_growth",
            training_script=WORKER,
            training_script_args=[],
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())
            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)

    def test_nway_gradient_full_sum(self):
        self._run_4accelerators()


if __name__ == "__main__":
    unittest.main()

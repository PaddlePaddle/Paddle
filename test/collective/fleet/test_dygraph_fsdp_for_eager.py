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
    TestMultipleAccelerators,
    get_cluster_from_args,
    get_devices,
    start_local_trainers,
    watch_local_trainers,
)

from paddle import base


class TestDygraphFSDP(TestMultipleAccelerators):
    # check dygraph fsdp for some functions.
    def test_dygraph_group_fsdp(self):
        self.run_mnist_2accelerators('dygraph_group_fsdp.py')

    # check dygraph fsdp + ep for some functions.
    def test_dygraph_group_fsdp_moe(self):
        self.run_mnist_2accelerators('dygraph_group_fsdp_moe.py')

    # check dygraph fsdp + ep with expert params sharded inside the ep group.
    def test_dygraph_group_fsdp_moe_sharding(self):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < 4
        ):
            self.skipTest(
                "moe_sharding_degree=2 comparison requires 4 GPUs, got "
                f"{base.core.get_cuda_device_count()}"
            )

        cluster, pod = get_cluster_from_args(get_devices('0,1,2,3'))
        procs = start_local_trainers(
            cluster,
            pod,
            training_script='dygraph_group_fsdp_moe.py',
            training_script_args=[],
        )
        while watch_local_trainers(procs, cluster.trainers_endpoints()):
            time.sleep(3)


if __name__ == "__main__":
    unittest.main()

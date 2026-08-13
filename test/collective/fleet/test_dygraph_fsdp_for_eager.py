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
    # All the cases below run on 4 cards, so this target is registered as
    # RUN_TYPE=EXCLUSIVE to get all the cards of the machine.
    def run_4accelerators(self, target_file_name, training_script_args=[]):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < 4
        ):
            self.skipTest(
                "bitwise loss comparison requires 4 GPUs, got "
                f"{base.core.get_cuda_device_count()}"
            )

        cluster, pod = get_cluster_from_args(get_devices('0,1,2,3'))
        procs = start_local_trainers(
            cluster,
            pod,
            training_script=target_file_name,
            training_script_args=training_script_args,
        )
        while watch_local_trainers(procs, cluster.trainers_endpoints()):
            time.sleep(3)

    # check dygraph fsdp for some functions.
    def test_dygraph_group_fsdp(self):
        self.run_4accelerators('dygraph_group_fsdp.py')

    # ep_degree=4 => moe_sharding_degree=1, expert params sharded on all cards;
    # ep_degree=2 => moe_sharding_degree=2, expert params sharded inside the
    # moe_sharding group, so their grads really go through reduce_scatter.
    def test_dygraph_group_fsdp_moe(self):
        for ep_degree in ['4', '2']:
            self.run_4accelerators('dygraph_group_fsdp.py', [ep_degree])


if __name__ == "__main__":
    unittest.main()

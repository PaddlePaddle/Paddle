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

import unittest

import collective.test_communication_api_base as test_base


class TestPipelineParallelP2pIssuedHook(test_base.CommunicationTestDistBase):
    def setUp(self):
        # `VPPFhenBInBalancedMemory` asserts `pp_degree > 2`, so the schedule
        # that raises `P2P_ISSUED` needs at least 4 ranks.
        super().setUp(num_of_devices=4, timeout=300, nnode=1)
        self._default_envs = {}
        # False exercises `_p2p_ops`, where the wait handles are deferred so
        # the hook can overlap the send/recv; True exercises `_batched_p2p_ops`,
        # where the location is raised without any deferral.
        self._changeable_envs = {"USE_BATCH_P2P_COMM": ["False", "True"]}

    def test_p2p_issued_hook(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "hybrid_parallel_pp_p2p_issued_hook.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()

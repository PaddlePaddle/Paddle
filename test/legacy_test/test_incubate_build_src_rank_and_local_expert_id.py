# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import build_src_rank_and_local_expert_id

logger = logging.getLogger(__name__)


class TestFusedCalculateAuxLoss(unittest.TestCase):
    def test_build_src_rank_and_local_expert_id(self):
        def orig_func(expert_num_global_list, num_local_experts):
            send_rank_cpu = np.concatenate(  # TOO SLOW!!! break every thing
                [
                    np.full([j], i // num_local_experts, dtype="int32")
                    for i, j in enumerate(expert_num_global_list)
                ],
                0,
            )
            local_expert_id_cpu = np.concatenate(
                [
                    np.full([j], i % num_local_experts, dtype="int32")
                    for i, j in enumerate(expert_num_global_list)
                ],
                0,
            )
            send_rank = paddle.to_tensor(send_rank_cpu)
            local_expert_id = paddle.to_tensor(local_expert_id_cpu)
            return send_rank, local_expert_id

        def fused_func(
            expert_num_global_tensor, expert_num_global, num_local_experts
        ):
            return build_src_rank_and_local_expert_id(
                expert_num_global_tensor, expert_num_global, num_local_experts
            )

        expert_num_global = np.random.randint(
            0, 512, size=[12 * 8], dtype="int32"
        )
        expert_num_global_tensor = paddle.to_tensor(
            expert_num_global, dtype="int64"
        )

        s1, l1 = orig_func(expert_num_global, 12)
        s2, l2 = fused_func(expert_num_global_tensor, expert_num_global, 12)
        assert ((s1 - s2) == 0).all(), (s1, s2)
        assert ((l1 - l2) == 0).all(), (l1, l2)


if __name__ == "__main__":
    unittest.main()

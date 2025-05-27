import os
import unittest

from op_test import convert_float_to_uint16
import random
import paddle.nn.functional as F

import paddle
import numpy as np
import random
import logging

import paddle
from paddle.nn.clip import _squared_l2_norm

from ernie_utils.top2_gate import (
    CalAuxLossFunctor,
    cal_aux_loss_func,
)
from paddle.incubate.nn.functional import build_src_rank_and_local_expert_id
from ernie_utils.moe_layer import fuse_logging

logger = logging.getLogger(__name__)



class TestFusedCalculateAuxLoss(unittest.TestCase):
    def test_build_src_rank_and_local_expert_id(self):
        def orig_func(expert_num_global_list, num_local_experts):
            send_rank_cpu = np.concatenate(  # TOO SLOW!!! break every thing
                [np.full([j], i // num_local_experts, dtype="int32") for i, j in enumerate(expert_num_global_list)],
                0,
            )
            local_expert_id_cpu = np.concatenate(
                [np.full([j], i % num_local_experts, dtype="int32") for i, j in enumerate(expert_num_global_list)],
                0,
            )
            send_rank = paddle.to_tensor(send_rank_cpu)
            local_expert_id = paddle.to_tensor(local_expert_id_cpu)
            return send_rank, local_expert_id

        def fused_func(expert_num_global_tensor, expert_num_global, num_local_experts):
            return build_src_rank_and_local_expert_id(
                expert_num_global_tensor, expert_num_global, num_local_experts
            )

        expert_num_global = np.random.randint(0, 512, size=[12 * 8],dtype="int32")
        expert_num_global_tensor = paddle.to_tensor(expert_num_global, dtype="int64")

        s1, l1 = orig_func(expert_num_global, 12)
        s2, l2 = fused_func(expert_num_global_tensor, expert_num_global, 12)
        assert ((s1 - s2) == 0).all(), (s1, s2)
        assert ((l1 - l2) == 0).all(), (l1, l2)



if __name__ == "__main__":
    unittest.main()
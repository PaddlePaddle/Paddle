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

import unittest
from collections import namedtuple
from functools import partial

from ernie_utils.moe_all_gather_layer import MOEAllGatherLayerV2

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import expand_modality_expert_id


def fused_gate_logits_process_ref(
    self, gate_logits_lm, gate_logits_mm, token_type_ids
):
    """process gatelogits"""
    top_k = self.k
    num_expert_per_rank_per_modality = (
        gate_logits_lm.shape[-1] // self.config.moe_world_size
    )

    @paddle.no_grad()
    def shift_ids(ids, modality_offset):
        # 现在认为所以模态的 expert 数都一样
        rank = ids // num_expert_per_rank_per_modality
        expert_id_in_rank = ids % num_expert_per_rank_per_modality
        return (
            rank * (num_expert_per_rank_per_modality * 2)
            + expert_id_in_rank
            + modality_offset * num_expert_per_rank_per_modality
        )

    if self.group_experts:
        gate_logits_lm = gate_logits_lm.reshape(
            [gate_logits_lm.shape[0], top_k, -1]
        )
        prob_lm = self.gate.act(gate_logits_lm)
        weight_lm, expert_id_lm = prob_lm.topk(k=1, axis=-1)
        weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
        expert_id_lm = expert_id_lm.reshape([gate_logits_lm.shape[0], -1])
        group_size = gate_logits_lm.shape[-1]
        scale = paddle.arange(0, top_k * group_size, group_size).unsqueeze(0)
        expert_id_lm = expert_id_lm + scale
    else:
        prob_lm = self.gate.act(gate_logits_lm)
        weight_lm, expert_id_lm = prob_lm.topk(k=top_k, axis=-1)
    if token_type_ids is not None:
        expert_id_lm = shift_ids(expert_id_lm, 0)
    expert_id_lm.stop_gradient = True
    lm_weight_and_expert_id = paddle.concat(
        [weight_lm, expert_id_lm.astype("float32")], -1
    )
    if token_type_ids is None:
        return (
            lm_weight_and_expert_id,
            prob_lm.reshape([prob_lm.shape[0], -1]),
            None,
        )

    prob_mm = self.gate.act(gate_logits_mm)
    weight_mm, expert_id_mm = prob_mm.topk(k=top_k, axis=-1)

    expert_id_mm = shift_ids(expert_id_mm, 1)
    expert_id_mm.stop_gradient = True

    mm_weight_and_expert_id = paddle.concat(
        [weight_mm, expert_id_mm.astype("float32")], -1
    )

    token_type_ids_float = token_type_ids[:, None].astype("float32")
    weight_and_expert = (
        (1 - token_type_ids_float) * lm_weight_and_expert_id
        + token_type_ids_float * mm_weight_and_expert_id
    )
    return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm


def test_expand_modality_expert_id():
    def expand_id_one(
        expert_id,
        num_expert_per_modality,
        k,
        group_size,
        modality_offset,
        is_group_expert,
    ):
        orig_shape = expert_id.shape
        expert_id = expert_id.reshape([-1])
        xid = paddle.arange(len(expert_id))
        if is_group_expert:
            eid = xid % k
            expert_id += eid * group_size

        rank = expert_id // num_expert_per_modality
        expert_id_in_rank = expert_id % num_expert_per_modality
        ret = (
            rank * (num_expert_per_modality * 2)
            + expert_id_in_rank
            + modality_offset * num_expert_per_modality
        )
        return ret.reshape(orig_shape)

    S, E, k = 100, 24, 3
    expert_id_mm = paddle.randint(0, 12, shape=[S, k])
    num_expert_per_rank_per_modality = E // 2 // 4
    group_size = E // 2 // k
    print(
        f"num_expert_per_rank_per_modality: {num_expert_per_rank_per_modality}"
    )
    fused = expand_modality_expert_id(
        expert_id_mm, num_expert_per_rank_per_modality, group_size, 1, True
    )

    nonfused = expand_id_one(
        expert_id_mm, num_expert_per_rank_per_modality, k, group_size, 1, True
    )
    # num_expert_per_rank_per_modality, group_size
    assert (fused == nonfused).all().item()

    Config = namedtuple("Config", ["moe_world_size"])
    Self = namedtuple(
        "Self",
        [
            "config",
            "k",
            "gate",
            "group_experts",
            "moe_statics",
            "use_correction_bias",
        ],
    )
    Gate = namedtuple("Gate", ["act"])
    fake_gate = Gate(act=partial(F.softmax, axis=-1))
    fake_self = Self(
        config=Config(
            moe_world_size=8,
        ),
        k=k,
        gate=fake_gate,
        moe_statics=None,
        use_correction_bias=False,
        group_experts=True,
    )

    fake_logits = paddle.randn([S, E])
    fake_logits_mm = paddle.randn([S, E])
    token_type_ids = paddle.randint(0, 2, shape=[S])
    w_and_e, prob_lm, prob_mm = (
        MOEAllGatherLayerV2.fused_gate_logits_process_fused(
            fake_self, fake_logits, fake_logits_mm, None
        )
    )
    w_and_e_ref, prob_lm_ref, prob_mm_ref = fused_gate_logits_process_ref(
        fake_self, fake_logits, fake_logits_mm, None
    )
    assert (prob_lm == prob_lm_ref).all().item()
    assert (w_and_e == w_and_e_ref).all().item()
    w, e = w_and_e_ref.chunk(2, axis=-1)


class Test_expand_modality_expert_id_API(unittest.TestCase):
    def test_dygraph(self):
        test_expand_modality_expert_id()


if __name__ == "__main__":

    unittest.main()

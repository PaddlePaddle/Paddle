# ruff: noqa: FA100
# !/usr/bin/env python3

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

"""
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: moe_layer_all_gather.py
@time: 2024/09/21 15:11:10
@Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""

from __future__ import annotations

import contextlib
import logging

import paddle
from paddle import nn
from paddle.incubate.nn.functional import expand_modality_expert_id

from .moe_layer import MOELayer

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group


def profile(_):
    """dumy profile"""
    return contextlib.nullcontext()


logger = logging.getLogger(__name__)

if False:
    try:
        from paddle_xpu_nn import moe_gate_dispatch as xpu_moe_gate_dispatch
    except ImportError:
        xpu_moe_gate_dispatch = None
        logger.warning("`xpu moe dispatch` not found")
else:
    pass


class MOEAllGatherLayer(MOELayer):
    """_summary_

    Args:
        MOELayer (_type_): _description_
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: list[nn.Layer],
        layer_idx,
        shared_experts: list[nn.Layer] | None = None,
        dense_experts: list[nn.Layer] | None = None,  # no use
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        enable_bpr: bool = False,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
    ):

        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            enable_logging,
            k,
            enable_bpr,
            all_to_all_dropout,
            group_experts,
            moe_statics,
        )


class MOEAllGatherLayerV2(MOEAllGatherLayer):
    """_summary_

    Args:
        MOELayer (_type_): _description_
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: list[nn.Layer],
        layer_idx,
        shared_experts: list[nn.Layer] | None = None,
        dense_experts: list[nn.Layer] | None = None,
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        enable_bpr: bool = False,
        enable_reverse_token_drop=False,
        all_to_all_dropout=0,
        group_experts=False,
        use_expert_out_alltoall=True,  #
        use_expert_alltoall_overlap=False,
        use_padding=True,
        dense_token_type=3,  # considered as dense tokens (no moe)
        moe_statics=None,
    ):
        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            dense_experts,
            group,
            recompute,
            enable_logging,
            k,
            enable_bpr,
            all_to_all_dropout,
            group_experts,
            moe_statics,
        )
        self.enable_reverse_token_drop = enable_reverse_token_drop
        self.is_allgather_moe_layer = True
        # assert self.gate.config.sequence_parallel
        world_size = self.gate.config.moe_world_size
        self.use_padding = use_padding

        # 全局 gate gather
        self.send_rank = None
        self.local_expert_id = None
        self.dense_token_type = dense_token_type
        self.dense_experts = dense_experts
        self.capacity_tensor = None
        self.use_expert_out_alltoall = use_expert_out_alltoall
        self.use_expert_alltoall_overlap = use_expert_alltoall_overlap
        logger.info(
            f"using MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "
            f"use_padding={use_padding}, use_expert_alltoall_overlap={use_expert_alltoall_overlap} "
            f"enable_reverse_token_drop={self.enable_reverse_token_drop}"
        )
        self.two = paddle.to_tensor(2, dtype=paddle.float32)
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

    def fused_gate_logits_process_fused(
        self, gate_logits_lm, gate_logits_mm, token_type_ids
    ):
        """process gatelogits w/ moe utils"""
        # top_k = 1 if isinstance(self.gate, SinkHornGateFused) else self.k
        top_k = self.k
        num_expert_per_rank_per_modality = (
            gate_logits_lm.shape[-1] // self.config.moe_world_size
        )
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape(
                [gate_logits_lm.shape[0], top_k, -1]
            )
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias:
                prob_lm_ = (
                    prob_lm
                    + self.moe_statics.e_score_correction_bias[0].detach()
                )
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)

        if self.use_correction_bias:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0])
                .unsqueeze(-1)
                .expand_as(expert_id_lm)
            )
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        # num_expert_per_modality == 0 时只执行 group-expert expand，不执行 multimodal-expand
        expert_id_lm = expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=(
                num_expert_per_rank_per_modality
                if (token_type_ids is not None and gate_logits_mm is not None)
                else 0
            ),
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = paddle.concat(
            [weight_lm, expert_id_lm.astype("float32")], -1
        )
        if token_type_ids is None or gate_logits_mm is None:
            return (
                lm_weight_and_expert_id,
                prob_lm.reshape([prob_lm.shape[0], -1]),
                None,
            )

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias:
            prob_mm_ = (
                prob_mm + self.moe_statics.e_score_correction_bias[1].detach()
            )
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
        if self.use_correction_bias:
            batch_idx = (
                paddle.arange(prob_lm_.shape[0])
                .unsqueeze(-1)
                .expand_as(expert_id_lm)
            )
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = paddle.concat(
            [weight_mm, expert_id_mm.astype("float32")], -1
        )
        weight_and_expert = paddle.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id,
            mm_weight_and_expert_id,
        )
        return (
            weight_and_expert,
            prob_lm.reshape([prob_lm.shape[0], -1]),
            prob_mm,
        )

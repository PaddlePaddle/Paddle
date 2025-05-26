# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: moe_layer_all_gather.py
@time: 2024/09/21 15:11:10
@Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释


"""
from typing import Any, Tuple, List, Dict, Optional, Callable
import itertools
from collections import defaultdict
import logging
import contextlib
import numpy as np
import inspect

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import framework
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.communication.group import Group

from .top2_gate import TopKGateFused, compute_optimal_transport
from paddle.incubate.tensor.manipulation import async_offload, async_reload

from .moe_layer import MOELayer, fuse_logging

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None


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
    try:
        import moe_ops
    except ImportError:
        moe_ops = None
        logger.warning("`moe-ops` not found, run " "`python3  src/ernie_core/ops/moe/setup.py  install` to install")
    try:
        import moe_ops_partial
    except ImportError:
        moe_ops_partial = None
        logger.warning(
            "`moe-ops-partial` not found, run " "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )
    try:
        import moe_ops_partial_nosoftmaxtopk
    except ImportError:
        moe_ops_partial_nosoftmaxtopk = None
        logger.warning(
            "`moe-ops-partial-nosoftmaxtopk` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )

    try:
        import moe_utils
    except ImportError:
        moe_utils = None
        logger.warning("`moe_utils` not found, run " "`python3  src/ernie_core/ops/moe/setup.py  install` to install")

class MOEAllGatherLayer(MOELayer):
    """_summary_

    Args:
        MOELayer (_type_): _description_
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        dense_experts: Optional[List[nn.Layer]] = None,  # no use
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
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        dense_experts: Optional[List[nn.Layer]] = None,
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
        dense_token_type=3,  # considerd as dense tokens (no moe)
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
            f"uisng MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "
            f"use_padding={use_padding}, use_expert_alltoall_overlap={use_expert_alltoall_overlap} "
            f"enable_reverse_token_drop={self.enable_reverse_token_drop}"
        )
        self.two = paddle.to_tensor(2, dtype=paddle.float32)
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)
 
    def fused_gate_logits_process_fused(self, gate_logits_lm, gate_logits_mm, token_type_ids):
        """process gatelogits w/ moe utils"""
        #top_k = 1 if isinstance(self.gate, SinkHornGateFused) else self.k
        top_k = self.k
        num_expert_per_rank_per_modality = gate_logits_lm.shape[-1] // self.config.moe_world_size
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape([gate_logits_lm.shape[0], top_k, -1])
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias:
                prob_lm_ = prob_lm + self.moe_statics.e_score_correction_bias[0].detach()
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)

        if self.use_correction_bias:
            batch_idx = paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        # num_expert_per_modality == 0 时只执行 group-expert expand，不执行 multimodal-expand
        expert_id_lm = moe_utils.expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=num_expert_per_rank_per_modality
            if (token_type_ids is not None and gate_logits_mm is not None)
            else 0,
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = paddle.concat([weight_lm, expert_id_lm.astype("float32")], -1)
        if token_type_ids is None or gate_logits_mm is None:
            return lm_weight_and_expert_id, prob_lm.reshape([prob_lm.shape[0], -1]), None

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias:
            prob_mm_ = prob_mm + self.moe_statics.e_score_correction_bias[1].detach()
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
        if self.use_correction_bias:
            batch_idx = paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = moe_utils.expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = paddle.concat([weight_mm, expert_id_mm.astype("float32")], -1)
        weight_and_expert = paddle.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id,
            mm_weight_and_expert_id,
        )
        return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

   

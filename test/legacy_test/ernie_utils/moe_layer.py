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

"""_summary_

Returns:
    _type_: _description_
"""
from __future__ import annotations

import logging
from collections import namedtuple
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group
try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None
try:
    from paddle.distributed import in_auto_parallel_align_mode
except:

    def in_auto_parallel_align_mode():
        """
        hack for paddlenlp develop branch.
        """
        return False


try:
    from bincount_ops import int_bincount
except ImportError:
    int_bincount = None

logger = logging.getLogger(__name__)

try:
    import moe_ops
except ImportError:
    moe_ops = None
    logger.warning(
        "`moe-ops` not found, run "
        "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
    )

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


class MOELayer(nn.Layer):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)

        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (paddle.nn.Layer):
            gate network
        expert (paddle.nn.LayerList):
            expert network, LayerList 长度是 per_device 上的 expert 数。
        group (paddle.ProgressGroup)
        recompute: 启用MOE内recomupte
    Returns:
        output
        combine_weight
        router-loss
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: list[nn.Layer],
        layer_idx,
        shared_experts: list[nn.Layer] | None = None,
        group: Group = None,
        recompute=False,
        enable_logging: bool = False,
        k=2,
        enable_bpr: bool = False,
        all_to_all_dropout=0,
        group_experts=False,
        moe_statics=None,
    ):
        """
        初始化MoE层。

        Args:
            gate (nn.Layer): 智能门控层，用于选择需要使用的专家。
            experts (List[nn.Layer]): 需要使用的专家列表。
            layer_idx (int): 当前MoE层的索引。
            group (Group): 分布式通信组。默认值为None。
            recompute (bool): 是否在每个训练迭代中重新计算MoE输出。默认值为False。
        """
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.recompute = recompute
        logger.info(f"using moe recompute={recompute}")
        for p in self.gate.parameters():
            p.is_gate = True
        if isinstance(experts, nn.LayerList):
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = experts
        self.shared_experts = shared_experts

        self.group = group
        self.k = k
        self.all_to_all_dropout = all_to_all_dropout
        self.enable_logging = enable_logging
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        if self.use_correction_bias:
            logger.info(
                f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}"
            )
            assert self.gate.config.moe_use_aux_free

        self.is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and group
            is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        is_dummy_moe = dist.get_world_size(group) == 1

        for p in experts.parameters():
            p.expert = not (self.is_mp_moe or is_dummy_moe)  # type: ignore
            p.no_sync = not (self.is_mp_moe or is_dummy_moe)
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe:
                p.is_distributed = True

        self.world_size = dist.get_world_size(self.group)
        # assert self.world_size > 1, f'moe-group not found, world_size {self.world_size}'
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = len(self.experts)
        self.dispatch_by_task = (
            hasattr(self.gate, "dispatch_by_task")
            and self.gate.dispatch_by_task
        )

        if self.dispatch_by_task:
            assert 0, "no supported, checkout earylier code"
            assert self.num_local_experts == 1

        ''' dummy skip
        if enable_bpr:
            logger.info("using BPR")
            prepost_process_buffer = {}
            self.input_preprocess = partial(
                bpr_preprocess, buffer=prepost_process_buffer
            )
            self.output_postprocess = partial(
                bpr_postprocess, buffer=prepost_process_buffer
            )
        else:
            self.input_preprocess = self.output_postprocess = None
        '''
        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

        self._rr_moe_gate_dispatch = None
        self._rr_moe_combine = None
        ''' dummy skip
        if self.config.use_recompute and self.config.skip_recompute_ops.get(
            "moe_gate_dispatch", False
        ):
            self._rr_moe_gate_dispatch = RefinedRcomputeMoEGateDispatch()
        if self.config.use_recompute and self.config.skip_recompute_ops.get(
            "moe_combine", False
        ):
            self._rr_moe_combine = RefinedRcomputeMoECombine()
        '''


def fuse_logging(gate_logits, combine_weights, token_type_ids):
    """fuse_logging"""
    with paddle.no_grad():
        gate_expert_per_token_type_0, gate_expert_per_token_type_1 = None, None
        gate_experts_per_token = None
        ce = moe_router_loss_ops.cal_cross_entropy_info(gate_logits).mean(0)
        if token_type_ids is not None:
            (
                gate_expert_per_token_type_0,
                gate_expert_per_token_type_1,
                gate_experts_per_token,
            ) = moe_router_loss_ops.cal_gate_experts_per_token_info(
                combine_weights, token_type_ids
            )
        else:
            gate_experts_per_token = paddle.count_nonzero(combine_weights) / (
                gate_logits.shape[0]
            )

        return (
            gate_expert_per_token_type_0,
            gate_expert_per_token_type_1,
            gate_experts_per_token,
            ce,
        )

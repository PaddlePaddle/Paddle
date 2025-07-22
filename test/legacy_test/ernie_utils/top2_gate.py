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
top2gate
"""

from __future__ import annotations

import logging
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.incubate.nn.functional import cal_aux_loss
from paddle.utils import unique_name

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量
try:
    import moe_router_loss_ops
except ImportError:
    moe_router_loss_ops = None

try:
    from custom_setup_ops import matmul_bwd
except ImportError:
    matmul_bwd = None

try:
    from bincount_ops import int_bincount
except ImportError:
    int_bincount = None

logger = logging.getLogger(__name__)


class CalAuxLossFunctor(paddle.autograd.PyLayer):
    """CalAuxLossFunctor"""

    @staticmethod
    def forward(
        ctx,
        gate_prob,
        dispatch_mask,
        tokens_mask,
        dispatch_tokens_mask,
        num_experts,
        use_group,
        moe_k,
        clip_min=1e-6,
    ):
        """forward"""
        if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
            tokens_mask = tokens_mask.astype(gate_prob.dtype)
        loss, seqlen_float, ce = cal_aux_loss(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            moe_k,
            clip_min,
        )
        '''
        ctx.save_for_backward(gate_prob, seqlen_float, ce)
        ctx.num_experts = num_experts
        ctx.use_group = use_group
        ctx.moe_k = moe_k
        '''
        return loss

    @staticmethod
    def backward(ctx, out_grad):
        """backward"""
        '''
        gate_prob, seqlen_float, ce = ctx.saved_tensor()
        num_experts = ctx.num_experts
        use_group = ctx.use_group
        moe_k = ctx.moe_k
        from paddle import _C_ops
        return _C_ops.cal_aux_loss_grad(
            out_grad, gate_prob, seqlen_float, ce, num_experts, use_group, moe_k
        )
        '''


def cal_aux_loss_func(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
    global_aux_loss=False,
    rank=None,
    group=None,
):
    """cal_aux_loss_func"""
    if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
        tokens_mask = tokens_mask.astype(gate_prob.dtype)

    scale = None
    if dispatch_tokens_mask is not None:
        seqlen_float = dispatch_tokens_mask.astype(gate_prob.dtype).sum()
        if (
            tokens_mask is not None
            and gate_prob.shape[0] != dispatch_tokens_mask.shape[0]
        ):
            scale = seqlen_float / paddle.clip(tokens_mask.sum(), min=1e-6)
    elif tokens_mask is not None:
        seqlen_float = tokens_mask.sum()
    else:
        seqlen_float = gate_prob.numel().astype(gate_prob.dtype) / num_experts
    seqlen_float = paddle.clip(seqlen_float, min=1e-6)
    if len(dispatch_mask.shape) == 2:
        dispatch_mask = dispatch_mask.sum(0)
    ce = dispatch_mask.astype(gate_prob.dtype).detach() / seqlen_float
    me = paddle.sum(gate_prob, axis=0) / seqlen_float
    # me = paddle.mean(gate_prob, axis=0)
    # ce = paddle.mean(dispatch_mask.cast("float32"), axis=0)
    if global_aux_loss:
        me_list, ce_list = [], []
        dist.all_gather(me_list, me, group=group)
        dist.all_gather(ce_list, ce, group=group)
        me_list[rank] = me
        ce_list[rank] = ce
        me = paddle.stack(me_list).mean(0)
        ce = paddle.stack(ce_list).mean(0)

    l_aux = paddle.sum(me * ce) * num_experts
    if use_group:
        l_aux = l_aux / moe_k
    if scale is not None:
        # 前向用局部me, 反向用全局me
        l_aux = l_aux + (scale - 1) * l_aux.detach()
    return l_aux


def masked_fill(x, mask, value):
    """
    将输入的Tensor中根据mask进行掩盖，并用value值替换。

    Args:
        x (Tensor): 输入的Tensor。
        mask (Tensor): 用于掩盖的布尔Tensor，其形状应与x相同。
        value (Union[float, int]): 需要替换的值。

    Returns:
        Tensor: 返回一个新的Tensor，其形状与x相同，并且根据mask和value进行掩盖和替换。

    """
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


@paddle.no_grad()
def compute_optimal_transport(
    M, r, c, lam=1.0, epsilon=1e-8, max_iters: int = 10
):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, _ = M.shape
    # P = (- lam * M).exp()
    # P /= P.sum()
    P = F.softmax(-M / lam)
    u = paddle.zeros(n, "float32")
    # normalize this matrix
    for _ in range(max_iters):
        if (u - P.sum(1)).abs().max() < epsilon:
            break
        u = P.sum(1)
        P *= (r / (u + 1e-8)).reshape((-1, 1))
        P *= (c / (P.sum(0) + 1e-8)).reshape((1, -1))
    P = paddle.where(~P.isnan(), P, paddle.zeros_like(P))
    return P, _


def cast_if_needed(x, dtype):
    """
    cast_if_needed
    """
    return x.cast(dtype) if x.dtype != dtype else x


class FusedGateDetachMatmul(paddle.autograd.PyLayer):
    """
    FusedGateDetachMatmul
    """

    @staticmethod
    def forward(ctx, x, w):
        """
        forward
        """
        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        return F.linear(
            cast_if_needed(x, ctx.dtype), cast_if_needed(w, ctx.dtype)
        )

    @staticmethod
    def backward(ctx, y_grad):
        """
        backward
        """
        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"
        x_g, w_g = matmul_bwd(
            cast_if_needed(x, ctx.dtype),
            cast_if_needed(w, ctx.dtype),
            y_grad,
            False,
            False,
        )
        return cast_if_needed(x_g, x.dtype), cast_if_needed(w_g, w.dtype)


def gate_detach_matmul(x, weight, use_fuse):
    """
    gate_detach_matmul
    """
    if use_fuse:
        return FusedGateDetachMatmul.apply(x, weight)
    else:
        x = cast_if_needed(x, paddle.float32)
        return F.linear(x, weight)


class Top2Gate(nn.Layer):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    def __init__(self, config, layer_idx: int, group, gate_weight=None) -> None:
        """
        初始化 MoE 层，包含参数初始化和一些其他功能。

        Args:
            layer_idx (int): 当前层的索引号。
            group: 分组名称。

        Returns:
            None: 不返回任何内容。
        """
        super().__init__()
        if False:
            try:
                from paddle_xpu.layers.nn import xpu_matmul

                self.xpu_matmul = xpu_matmul()
            except ImportError:
                self.xpu_matmul = None

        self.config = config
        self.fuse_gate_detach_matmul = config.fuse_gate_detach_matmul
        if self.fuse_gate_detach_matmul:
            assert matmul_bwd is not None, "matmul_bwd is not supported"

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = (
            sum(config.moe_num_experts)
            if config.multimodel_experts
            else config.moe_num_experts
        )  # paddle.to_tensor(config.moe_num_experts, dtype="float32").sum()

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx
        self.global_aux_loss = config.global_aux_loss
        if self.global_aux_loss:
            self.rank = dist.get_rank(self.group)

        self.sinkhorn_2gate = config.sinkhorn_2gate
        self.sinkhorn_temp = config.sinkhorn_temp
        self.use_token_type_bias = config.moe_use_token_type_bias
        self.use_correction_bias = config.moe_use_aux_free

        if config.moe_gate_act == "softmax":
            self.act = partial(F.softmax, axis=-1)  # [S,E]
        elif config.moe_gate_act == "sigmoid":
            self.act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")
        self.no_jitter = True
        self.expert_drop = False
        self.eye_matrix = None
        self.eye_matrix_size = None
        self.enable_logging = config.moe_logging
        self.norm_gate_logits = config.moe_norm_gate_logits
        self.one = paddle.ones([], dtype="float32")

        self.moe_aux_loss_lambda = paddle.to_tensor(
            config.moe_aux_loss_lambda, dtype="float32"
        )
        self.moe_z_loss_lambda = paddle.to_tensor(
            config.moe_z_loss_lambda, dtype="float32"
        )
        self.moe_orthogonal_loss_lambda = paddle.to_tensor(
            config.moe_orthogonal_loss_lambda, dtype="float32"
        )
        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)
        if self.moe_z_loss_lambda.ndim == 0:
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.unsqueeze(0)
        if self.moe_orthogonal_loss_lambda.ndim == 0:
            self.moe_orthogonal_loss_lambda = (
                self.moe_orthogonal_loss_lambda.unsqueeze(0)
            )

        self.experts_type_ids = None
        if config.moe_orthogonal_loss_lambda:
            if hasattr(fleet.fleet, "_user_defined_strategy"):
                strategy = fleet.fleet._user_defined_strategy
                sharding_configs = strategy.hybrid_configs["sharding_configs"]
                pp_config = strategy.hybrid_configs["pp_configs"]
                assert (
                    not sharding_configs.comm_overlap
                    and not pp_config.sharding_comm_overlap
                ), "orthogonal loss will cause twice gradient accumulate, will break pp/sharding overlap"

        self.eps = paddle.to_tensor([1e-12], dtype="float32")
        if config.multimodel_experts:
            if config.moe_use_hard_gate:
                self.num_experts_list = []
                self.experts_type_mask = []
                # hard-gate + group_experts 需要对gate_logits不同部分分开计算
                experts_ids = paddle.zeros(
                    [sum(self.num_experts)], dtype="int64"
                ).reshape([config.moe_world_size, -1])
                offset = 0
                for i, expert_num in enumerate(self.num_experts):
                    experts_ids[
                        :, offset : offset + expert_num // config.moe_world_size
                    ] = i
                    offset += expert_num // config.moe_world_size
                self.experts_type_ids = experts_ids.reshape([-1])
                logger.info(
                    f"use moe_use_hard_gate, experts_ids: {self.experts_type_ids}"
                )
                for i, expert_num in enumerate(self.num_experts):
                    self.experts_type_mask.append(
                        self.experts_type_ids == i,
                    )
                    self.num_experts_list.append(expert_num)
            else:
                # 非group_experts, 依赖token_type_bias实现hard-gate能力。
                assert (
                    not config.moe_group_experts
                ), "group_experts must use hard_gate when multimodel_experts is True"
        else:
            self.num_experts_list = [self.num_experts]
        if gate_weight is not None:
            self.weight = gate_weight
            assert (
                not self.config.moe_use_token_type_bias
            ), "gate_weights is from outside, token_type_bias can't be used"
            logger.info("moe use gate_weight from outside")
            # 强制在amp下任使用fp32精度
            self._cast_to_low_precision = False  # 兼容develop分支paddle
            self._cast_to_low_precision = False
        else:
            self._create_gate_parameter()
        logger.info(
            f"{config.moe_gate}: w/ capacity: {self.cap} experts:{self.num_experts} "
            f"use_token_type_bias:{self.use_token_type_bias} gate_act:{config.moe_gate_act} "
            f"norm_gate_logits={self.norm_gate_logits} use_correction_bias={self.use_correction_bias}"
        )

    def _create_gate_parameter(self):
        """
        创建参数权重。

        Args:
            None

        Returns:
            weight (Parameter): 创建的参数权重。

        """
        if self.config.multimodel_experts:
            # support setting lambda for each expert group
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_orthogonal_loss_lambda = (
                self.moe_orthogonal_loss_lambda.expand(len(self.num_experts))
            )

            for i, num_experts in enumerate(self.num_experts):
                if i == 1:
                    with paddle.utils.unique_name.guard(
                        f"mm_gate_{self.layer_idx}_"
                    ):
                        p = self.create_parameter(
                            shape=[self.model_dim, num_experts],
                            dtype="float32",
                            attr=paddle.ParamAttr(
                                name=unique_name.generate("moe_gate")
                            ),
                        )
                else:
                    p = self.create_parameter(
                        shape=[self.model_dim, num_experts],
                        dtype="float32",
                        attr=paddle.ParamAttr(
                            name=unique_name.generate("moe_gate")
                        ),
                    )
                p.expert_type = f"expert_type_{i}"
                self.add_parameter(
                    (
                        "weight" if i == 0 else f"weight_{i}"
                    ),  # 为了对齐原 state-dict，第一个 gate-weight 不改名.
                    p,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.model_dim, self.num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(
                    name=unique_name.generate("moe_gate")
                ),  # 特殊处理，有利于热启 dense-ckpt
            )
            logger.info(f"moe-Gate, {self.weight}")

        if self.use_token_type_bias:
            if self.config.multimodel_experts:
                assert (
                    not self.config.moe_use_hard_gate
                ), "multimodel_experts with hard_gate is not support token_type_bias."
            num_experts = (
                sum(self.num_experts)
                if self.config.multimodel_experts
                else self.num_experts
            )
            bias_type_num = (
                len(self.num_experts) if self.config.multimodel_experts else 1
            )
            self.bias = self.create_parameter(
                shape=[bias_type_num, num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(
                    name=unique_name.generate("moe_gate_bias"),
                    initializer=paddle.nn.initializer.Assign(
                        np.zeros([bias_type_num, num_experts])
                    ),
                ),  # 特殊处理，有利于热启 dense-ckpt
            )
            logger.info(f"using token type bias, bias: {self.bias},")
        # 强制在amp下任使用fp32精度
        self._cast_to_low_precision = False  # 兼容develop分支paddle
        self._cast_to_low_precision = False

    def get_gate_weight(self, transform_weight):
        """
        在`multimodel_experts` 的情况下，将多个 weights merge 成一个整体
        transform_weight: bool, 按照 local-expert id 将 多模态 weight 交叠
        """
        if not self.config.multimodel_experts:
            return self.weight
        if not transform_weight:
            return paddle.concat(
                [
                    getattr(self, "weight" if i == 0 else f"weight_{i}")
                    for i in range(len(self.num_experts))
                ],
                -1,
            )
        weight = paddle.zeros(
            [
                self.model_dim,
                self.config.moe_world_size,
                sum(self.num_experts) // self.config.moe_world_size,
            ],
            dtype="float32",
        )
        offset = 0
        for i, num_experts in enumerate(self.num_experts):
            weight[
                :,
                :,
                offset : offset + num_experts // self.config.moe_world_size,
            ] = getattr(self, "weight" if i == 0 else f"weight_{i}").reshape(
                [self.model_dim, self.config.moe_world_size, -1]
            )
            offset += num_experts // self.config.moe_world_size
        weight = weight.reshape([self.model_dim, -1])

        return weight

    def forward(
        self,
        input: Tensor,
        token_type_ids: Tensor = None,
        transform_weight: bool = True,  # [seq]
        correction_bias: Tensor = None,  # [seq]
    ) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor[Seq, Dim], hidden-states of layer
            token_type_ids: paddle.Tensor[Seqw], token_type_ids of input
            transform_weight: bool, when using multimodal experts, perform `self.get_gate_weight` if specified
        Returns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        num_experts = (
            sum(self.num_experts)
            if self.config.multimodel_experts
            else self.num_experts
        )
        orig_dtype = input.dtype
        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            if False:
                assert not self.fuse_gate_detach_matmul, "not supported on XPU"
                input_32 = input.cast("float32")
                logits = self.xpu_matmul(
                    input_32,
                    weight,
                    training=self.training,
                )
            else:
                logits = gate_detach_matmul(
                    input, weight, self.fuse_gate_detach_matmul
                )

            if self.use_token_type_bias:
                assert token_type_ids is not None
                bias = self.bias[token_type_ids]  # [seq]
                # logger.info(f"adding bias: {bias}")
                logits = logits + bias
            (
                capacity,
                dispatch_mask,
                combine_weights,
                scatter_index,
                l_aux,
                l_zloss,
            ) = self.top2_gating(logits, correction_bias=correction_bias)
            orthogonal_loss = self._cal_orthogonal_loss()
            router_loss = (
                l_aux * self.moe_aux_loss_lambda
                + l_zloss * self.moe_z_loss_lambda
                + orthogonal_loss * self.moe_orthogonal_loss_lambda
            )
            router_loss.stop_gradient = False

        combine_weights = combine_weights.cast(orig_dtype)
        return (
            capacity,
            dispatch_mask,
            combine_weights,
            scatter_index,
            router_loss,
            logits,
        )

    def get_capacity(self, num_tokens, cap_factor=None):
        """
        return capacity
        """
        num_experts = (
            sum(self.num_experts)
            if self.config.multimodel_experts
            else self.num_experts
        )
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.cap[0]
            elif num_tokens < num_experts:  # seqlen < num_expert
                cap = self.cap[2]
            else:
                cap = self.cap[1]
        # capacity = 2S/E
        capacity = int(cap * num_tokens // num_experts)
        assert (
            capacity > 0
        ), f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity

    def top2_gating(self, logits, cap=None, correction_bias=None):
        """
        Args:
            logits: 形状为[batch, vocab_size]的logits，用于计算top2 gate。
            cap[Optional]: capacity-factor, if none, read from config
            correction_bias[Optional]: used for aux-free router

        Returns:
            tuple:
                - capacity: 每个token可分发的最大数量。
                - dispatch_masks: 用于dispatching的mask。第一个元素是第一类token的mask；第二个元素是第二类token的mask。
                - combine_weights：用于combining的权重。第一个元素是第一类token的权重；第二个元素是第二类token的权重。
                - scatter_indexes: 用于scattering的索引。第一个元素是第一类token的索引；第二个元素是第二类token的索引。
                - loss_aux: aux loss。
                - loss_z: z loss。
        """
        # logger.info(f'gate-input: {logits}')
        l_zloss = self._cal_z_loss(logits)
        gates = self.act(logits)

        # gates has shape of SE
        assert logits.ndim == 2, logits.shape
        num_tokens = gates.shape[0]
        num_experts = gates.shape[1]
        # capacity = 2S/E
        capacity = self.get_capacity(logits.shape[0], cap)

        # Create a mask for 1st's expert per token
        score_for_argmax = (
            gates + correction_bias.unsqueeze(0)
            if correction_bias is not None
            else gates
        )
        indices1_s = paddle.argmax(score_for_argmax, axis=1)
        mask1 = F.one_hot(indices1_s, num_classes=num_experts).cast(
            paddle.int64
        )  # [0,1]

        l_aux = self._cal_aux_loss(
            gates, mask1.sum(axis=0), self.num_experts_tensor
        )
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        if self.training and not self.no_jitter:
            gumbels = (
                -paddle.empty_like(
                    logits,
                )
                .exponential_()
                .log()
            )  # ~Gumbel(0,1)
            logits_w_noise = logits + gumbels
        else:
            logits_w_noise = logits

        logits_except1 = masked_fill(
            logits_w_noise, mask1.cast(paddle.bool), float("-inf")
        )
        score_for_argmax = (
            self.act(logits_except1) + correction_bias.unsqueeze(0)
            if correction_bias is not None
            else logits_except1
        )
        indices2_s_original = paddle.argmax(score_for_argmax, axis=1)

        if self.training and self.sinkhorn_2gate:
            r = paddle.ones(num_tokens, "float32") / num_tokens
            # c = paddle.ones(num_experts, "float32") / num_experts
            # 非均匀c
            c = capacity - mask1.cast("float32").sum(0)
            c = paddle.maximum(c, paddle.zeros_like(c))
            c /= c.sum()

            pi, _ = compute_optimal_transport(
                -logits_except1.cast("float32").detach(),
                r,
                c,
                lam=self.sinkhorn_temp,
            )
            pi = masked_fill(pi, mask1.cast(paddle.bool), float("-inf"))
            indices2_s = paddle.argmax(pi, axis=1)
        else:
            indices2_s = indices2_s_original

        mask2 = F.one_hot(indices2_s, num_classes=self.num_experts).cast(
            paddle.int64
        )

        # Compute locations in capacity buffer
        locations1 = (
            paddle.cumsum(mask1, axis=0) - 1
        )  # [0,1,1,0,1,0,0] -> [0,0,0,0,1,1,1,]
        locations2 = paddle.cumsum(mask2, axis=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += paddle.sum(mask1, axis=0, keepdim=True)

        # Remove locations outside capacity from mask
        mask1 *= (locations1 < capacity).cast(paddle.int64)  # [0,1,1,0,0,0,0]
        mask2 *= (locations2 < capacity).cast(paddle.int64)

        # Store the capacity location for each token
        locations1_s = paddle.sum(locations1 * mask1, axis=1)
        locations2_s = paddle.sum(locations2 * mask2, axis=1)

        # Normalize gate probabilities
        mask1_float = mask1.cast(paddle.float32)
        mask2_float = mask2.cast(paddle.float32)
        gates1_s = (gates * mask1_float).sum(axis=-1)
        gates2_s = (gates * mask2_float).sum(axis=-1)
        # logger.info(f'gates1_s:{gates1_s} gates2_s:{gates2_s} logits:{logits}')

        if self.norm_gate_logits:
            denom_s = gates1_s + gates2_s  # [0.2, 0.3]
            # Avoid divide-by-zero
            denom_s = paddle.clip(denom_s, min=1e-6)
            gates1_s /= denom_s
            gates2_s /= denom_s
        if self.training and self.expert_drop:
            # log.debug(gates2_s)
            gates2_s = paddle.where(
                2 * gates2_s < paddle.rand_like(gates2_s),
                paddle.zeros_like(gates2_s),
                gates2_s,
            )

        # Calculate combine_weights and dispatch_mask
        gates1 = gates1_s.unsqueeze(1) * mask1_float
        gates2 = gates2_s.unsqueeze(1) * mask2_float

        expert1_index = paddle.argmax(gates1, -1)
        combine1_weight = paddle.max(gates1, -1, keepdim=True)
        scatter1_index = expert1_index * capacity + locations1_s
        scatter1_index = scatter1_index.cast("int64")
        dispatch1_mask = combine1_weight.cast(paddle.bool).detach()

        expert2_index = paddle.argmax(gates2, -1)
        combine2_weight = paddle.max(gates2, -1, keepdim=True)
        scatter2_index = expert2_index * capacity + locations2_s
        scatter2_index = scatter2_index.cast("int64")
        dispatch2_mask = combine2_weight.cast(paddle.bool).detach()
        # logger.info(f'expert-id: {expert1_index} vs {expert2_index}, mask:{mask1_float} vs {mask2_float}')

        return (
            capacity,
            paddle.concat((dispatch1_mask, dispatch2_mask), 1),
            paddle.concat((combine1_weight, combine2_weight), 1),
            paddle.stack((scatter1_index, scatter2_index), 1),
            l_aux,
            l_zloss,
        )

    def _cal_aux_loss(
        self,
        gate_prob,
        dispatch_mask,
        num_experts=None,
        use_group=None,
        tokens_mask=None,
        dispatch_tokens_mask=None,
    ):
        """
        计算辅助损失

        Args:
            gate_prob (paddle.Tensor[local_seq, num_experts]):
            dispatch_mask (paddle.Tensor[num_experts]): 每个 expert 被分配的 token 数（不考虑 token drop)
            tokens_mask (paddle.Tensor[Seq]): 每个 MP 内 token-type-id
            dispatch_tokens_mask (paddle.Tensor): AllGather 后的`tokens_mask`
        Returns:
            paddle.Tensor: 辅助损失值。

        """
        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(
                        k=self.config.moe_k, axis=-1
                    )
                    if int_bincount is not None:
                        dispatch_mask = int_bincount(
                            top_idx, 0, gate_prob.shape[-1], paddle.int64
                        )
                    else:
                        mask = paddle.zeros_like(
                            gate_prob_this_modality
                        ).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)
                        dispatch_mask = paddle.sum(
                            mask.cast(paddle.int64), axis=0
                        )
                else:
                    dispatch_mask = paddle.zeros(
                        gate_prob.shape[-1], dtype="int64"
                    )
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=self.config.moe_k, axis=-1)
                if int_bincount is not None:
                    dispatch_mask = int_bincount(
                        top_idx, 0, gate_prob.shape[-1], paddle.int64
                    )
                else:
                    mask = paddle.zeros_like(gate_prob).put_along_axis(
                        top_idx, paddle.to_tensor(1.0), axis=1
                    )
                    dispatch_mask = paddle.sum(mask.cast(paddle.int64), axis=0)

        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        if (
            moe_router_loss_ops is not None
            and (tokens_mask is None or len(tokens_mask.shape) == 1)
            and (
                tokens_mask is None
                or tokens_mask.shape[0] == gate_prob.shape[0]
            )
            and (gate_prob.shape[0] >= gate_prob.shape[1])
            and (not self.global_aux_loss)
            and (gate_prob.dtype == paddle.float32)
        ):
            return CalAuxLossFunctor.apply(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                num_experts,
                use_group,
                self.config.moe_k,
                clip_min=1e-6,
            )
        else:
            return cal_aux_loss_func(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                num_experts,
                use_group,
                self.config.moe_k,
                self.global_aux_loss,
                self.rank if self.global_aux_loss else None,
                self.group if self.global_aux_loss else None,
            )


class TopKGateFused(Top2Gate):
    """doc"""

    def forward(
        self,
        input: Tensor,
        token_type_ids=None,
        transform_weight=True,
    ) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """
        Args:
            input: paddle.Tensor, hidden-states of layer
            token_type_ids: paddle.Tensor[Seqw], token_type_ids of input
            transform_weight: bool, when using multimodal experts, perform `self.get_gate_weight` if specified
        Returns:
            paddle.Tensor [Seq, Expert, Capacity]: float32, combine weights
            paddle.Tensor [Seq, Expert, Capacity]: bool, dispatch mask
            Tuple[paddle.Tensor]: `GateOutput`
        """
        capacity = self.get_capacity(input.shape[0])
        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            if False:
                assert not self.fuse_gate_detach_matmul, "not supported on XPU"
                input_32 = input.cast("float32")
                logits = self.xpu_matmul(
                    input_32,
                    weight,
                    training=self.training,
                )
            else:
                logits = gate_detach_matmul(
                    input, weight, self.fuse_gate_detach_matmul
                )
            if self.use_token_type_bias:
                assert token_type_ids is not None
                assert (
                    token_type_ids.max() < self.bias.shape[0]
                ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
                bias = self.bias[token_type_ids]  # [seq]
                logits = logits + bias
            orthogonal_loss = None
            # 正交 loss 拿到 moe-layer 里去计算
            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False

        return logits, capacity, router_loss


class DeepEPTop2Gate(TopKGateFused):
    """DeepEPTop2Gate"""

    def forward(
        self,
        input,
        transform_weight=True,
        global_gate_mask=None,
        input_ids=None,
    ):
        """forward"""

        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            logits = gate_detach_matmul(
                input, weight, self.fuse_gate_detach_matmul
            )

        if global_gate_mask is not None:
            logits = logits + global_gate_mask
        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        return logits, router_loss

    def _cal_aux_loss(self, gates, dispatch_mask, input_ids=None):
        """
        Calculate auxiliary loss

        Args:
            gates (paddle.Tensor): Represents the output probability of each expert.
                The shape is [seq_len, num_experts]
            dispatch_mask: (paddle.Tensor): Represents the number of tokens for each expert.
                The shape is [num_experts]
            topk_indices:
        Returns:
            paddle.Tensor: The value of auxiliary loss.

        """
        assert (
            len(gates.shape) == 2
        ), "gates.shape must be [sequence_length, num_experts]"
        if input_ids is not None:
            # has_padding = (input_ids == 0).any()
            assert (
                input_ids.shape[0] == gates.shape[0]
            ), f"check input_ids shape {input_ids.shape}"
            valid_mask = (input_ids != 0).astype(paddle.float32)
            seqlen_float = valid_mask.sum().item()
            gates = gates * valid_mask.unsqueeze(-1)
        else:
            seqlen_float = float(gates.shape[0])
        me = paddle.sum(gates, axis=0) / seqlen_float
        ce = dispatch_mask.astype(gates.dtype).detach() / seqlen_float

        if self.global_aux_loss:
            me_list, ce_list = [], []
            dist.all_gather(me_list, me, group=self.group)
            dist.all_gather(ce_list, ce, group=self.group)

            me_list[self.rank] = me
            ce_list[self.rank] = ce
            me = paddle.stack(me_list).mean(0)
            ce = paddle.stack(ce_list).mean(0)
        if seqlen_float == 0:
            return paddle.to_tensor(0.0)
        aux_loss = paddle.sum(me * ce) * float(self.num_experts)
        return aux_loss

    def _cal_z_loss(self, logits) -> paddle.Tensor:
        """
        Calculate the z loss.

        Args:
            logits (paddle.Tensor): Model output. The shape is [batch_size, num_experts].

        Returns:
            paddle.Tensor: The z loss value.
        """
        l_zloss = paddle.logsumexp(logits, axis=1).square().mean()
        return l_zloss

    def _cal_orthogonal_loss(self) -> paddle.Tensor:
        """Gate weight orthogonal loss.

        Returns:
            Paddle.Tensor: orthogonal loss
        """
        weight = F.normalize(self.weight, axis=0)
        orthogonal_loss = paddle.mean(
            paddle.square(
                paddle.matmul(weight.T, weight) - paddle.eye(self.num_experts)
            )
        )
        return orthogonal_loss

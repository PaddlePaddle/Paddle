# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#
# The file has been adapted from the file:
#     https://github.com/laekov/fastmoe/blob/master/fmoe/layers.py
#     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
# We retain the following license from the original files:
#     Copyright 2021, Jiaao He. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License").

import os

import numpy as np

import paddle
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed.utils.moe_utils import global_gather, global_scatter
from paddle.distributed.utils.nccl_utils import check_nccl_version_for_p2p
from paddle.framework import in_dynamic_mode
from paddle.incubate.distributed.fleet import recompute_hybrid

from .gate import BaseGate, GShardGate, NaiveGate, SwitchGate
from .utils import count_by_gate


def _local_scatter(inp, pos):
    if pos.shape != [0]:
        inp_buf = paddle.index_select(inp, pos, 0)
    else:
        inp_buf = paddle.empty([0, inp.shape[1]], dtype=inp.dtype)
    return inp_buf


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    if pos.shape != [0]:
        origin_dtype = inp.dtype
        inp = paddle.cast(inp, dtype="float32")
        inp_buf = paddle.scatter(
            paddle.zeros(
                shape=[out_batch_size, inp.shape[-1]], dtype="float32"
            ),
            pos,
            inp,
            overwrite=True,
        )
        inp_buf = paddle.cast(inp_buf, dtype=origin_dtype)
    else:
        inp_buf = paddle.zeros([out_batch_size, inp.shape[-1]], dtype=inp.dtype)
    return inp_buf


def _all_gather(tensor, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return

    if in_dynamic_mode():
        group = (
            paddle.distributed.collective._get_default_group()
            if group is None
            else group
        )
        tensor_shape = list(tensor.shape)
        tensor_shape[0] *= group.nranks
        out = paddle.empty(tensor_shape, tensor.dtype)

        task = group.process_group.all_gather(tensor, out)
        task.wait()
        return out
    else:
        ring_id = 0 if group is None else group.id
        nranks = (
            paddle.distributed.collective._get_global_group().nranks
            if group is None
            else group.nranks
        )
        return paddle._legacy_C_ops.all_gather(
            tensor,
            'ring_id',
            ring_id,
            'nranks',
            nranks,
        )


class MoEScatter(PyLayer):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        pos,
        local_expert_count,
        global_expert_count,
        fwd_batch_size,
        world_size,
        group=None,
    ):
        local_input_buf = _local_scatter(inp, pos)
        if world_size > 1:
            global_input_buf = global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                group=group,
            )
        else:
            global_input_buf = local_input_buf

        ctx.moe_args = inp.shape[0], world_size, group

        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, grad):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensor()
        (inp_batch_size, world_size, group) = ctx.moe_args

        if world_size > 1:
            local_grad_in = global_gather(
                grad, local_expert_count, global_expert_count, group=group
            )
        else:
            local_grad_in = grad
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None


class MoEGather(PyLayer):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MoEScatter.
    """

    @staticmethod
    def forward(
        ctx,
        global_output_buf,
        pos,
        local_expert_count,
        global_expert_count,
        local_batch_size,
        world_size,
        group=None,
    ):
        if world_size > 1:
            local_output_buf = global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                group=group,
            )
        else:
            local_output_buf = global_output_buf
        output = _local_gather(
            local_output_buf, pos, local_batch_size, maybe_overlap=False
        )

        ctx.moe_args = (global_output_buf.shape[0], world_size, group)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensor()
        fwd_batch_size, world_size, group = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out, pos)
        if world_size > 1:
            global_grad_out_buf = global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                group=group,
            )
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None


class AllGather(PyLayer):
    r"""
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, inp, group=group)
        output = paddle.concat(tensor_list, axis=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return paddle.slice(
            grad_out, axes=[0], starts=[rank * dim0], ends=[(rank + 1) * dim0]
        )


class Slice(PyLayer):
    r"""
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = paddle.slice(
            inp, axes=[0], starts=[batch_start], ends=[batch_end]
        )
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        return _all_gather(grad_out, group=group)


def prepare_forward(gate, num_expert, world_size, moe_group):
    pos, local_expert_count, global_expert_count = count_by_gate(
        gate, num_expert, world_size, group=moe_group
    )
    with paddle.no_grad():
        fwd_expert_count = global_expert_count.reshape_(
            [world_size, num_expert]
        ).sum(axis=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    )


class MoELayer(nn.Layer):
    """MoE Layer
    Args:
        d_model (int): Model dimension.
        experts (nn.LayerList): Expert networks list.
        gate (dict|NaiveGate|SwitchGate|NaiveGate):

            - If gate is a dict:
              gate is a gate network config, containing 2 keys:
              `type` (str) value can be: "naive", "gshard", "switch" or None, default is "gshard".
              `top_k` (int) Default value is 2.
            else gate is an instance of NaiveGate|SwitchGate|NaiveGate:

        moe_group: moe group for experts communication.
        mp_group: mp group for mp communication.
        recompute_interval (int, optional): Whether to use recompute, default 0, means to disable recompute.
        recompute_ctx (dict, optional): The context for recompute, if recompute_interval > 1, recompute_ctx must be given.

    Examples:

        .. code-block:: python

            >>> # doctest: +SKIP('Until Distributed move successfully, just skip it')
            >>> from paddle.nn import layer, LayerList
            >>> from paddle.distributed.moe import MoElayer
            >>> from paddle.distributed.collective import Group
            >>> from paddle.distributed import fleet

            >>> moe_group = Group(fleet.worker_index(),
            ...                   0,
            ...                   list(range(fleet.worker_num())))
            >>> mp_group = None

            >>> num_experts=8
            >>> dim_feedforward=512
            >>> d_model=8
            >>> top_k=2

            >>> class ExpertLayer(Layer):
            ...     def __init__(self, d_model, d_hidden, name=None,rank=0, windex = 0, num_expert=1):
            ...         super().__init__()
            ...         self.htoh4 = nn.Linear(d_model, d_hidden)
            ...         self.h4toh = nn.Linear(d_hidden, d_model)

            ...     def forward(self, x):
            ...         x = self.htoh4(x)
            ...         x = self.h4toh(x)
            ...         return x

            >>> gate_config = {
            ...         "type": "gshard",
            ...         "top_k": top_k,
            ... }

            >>> experts_list = LayerList()
            >>> for expi in range(num_experts):
            ...     exp_layer = ExpertLayer(d_model, dim_feedforward // top_k, windex=expi, num_expert=num_experts)
            ...     experts_list.append(exp_layer)

            >>> moeLayer = MoELayer(d_model = d_model,
            ...                     experts=experts_list,
            ...                     gate=gate_config,
            ...                     moe_group=moe_group,
            ...                     mp_group=mp_group,
            ...                     recompute_interval=0)

    """

    def __init__(
        self,
        d_model,
        experts,
        gate=None,
        moe_group=None,
        mp_group=None,
        recompute_interval=0,
        recompute_ctx=None,
    ):
        super().__init__()

        self.recompute_ctx = recompute_ctx

        if gate is None:
            gate = {}

        assert isinstance(
            gate, (dict, BaseGate)
        ), "gate config' type must be dict or an instance of BaseGate"
        # only support mp/dp
        self.group = moe_group

        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = len(experts)
        self.recompute_interval = recompute_interval
        assert experts is not None
        self.experts = experts

        if (
            self.world_size > 1
            and os.getenv("PADDLE_DISTRI_BACKEND", None) != "xccl"
        ):
            check_nccl_version_for_p2p()

        self.mp_group = mp_group
        self.d_model = d_model
        if isinstance(gate, dict):
            self.top_k = gate.get("top_k", 2)
            gate = gate.get("type", "gshard")
            if gate == "naive" or gate is None:
                gate = NaiveGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k,
                )
            elif gate == "gshard":
                gate = GShardGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k,
                    group=self.group,
                )
            elif gate == "switch":
                gate = SwitchGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k,
                    group=self.group,
                )
            else:
                raise AssertionError(
                    f"We only support naive gate,                                 gshard gate and switch gate,                                 but you choose {gate} gate."
                )
        elif isinstance(gate, NaiveGate):
            self.top_k = gate.top_k
        elif isinstance(gate, BaseGate):
            raise TypeError(f"Unimplemented gate type: {type(gate)}")
        else:
            raise TypeError("gate's type must be either dict or moe.BaseGate")
        self.gate = gate

    def forward(self, inp):
        # inp shape: b * s * m
        assert len(inp.shape) == 3
        origin_shape = inp.shape
        inp = inp.reshape_([-1, origin_shape[2]])

        mp_rank = 0
        mp_size = 1
        if self.mp_group is not None:
            mp_rank = self.mp_group.rank
            mp_size = self.mp_group.nranks
        if mp_size > 1:
            inp = Slice.apply(inp, mp_rank, mp_size, self.mp_group)
        value, gate = self.gate(inp)

        (
            pos,
            local_expert_count,
            global_expert_count,
            fwd_expert_count,
            fwd_batch_size,
        ) = prepare_forward(gate, self.num_expert, self.world_size, self.group)

        topk = 1
        if len(gate.shape) == 2:
            topk = gate.shape[1]

        if pos.shape != [0]:
            temp_pos = pos // topk
        else:
            temp_pos = pos
        assert topk == self.top_k

        x = MoEScatter.apply(
            inp,
            temp_pos,
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            self.world_size,
            self.group,
        )

        d_model = self.d_model

        def experts_fwd(x, fwd_expert_count, experts):
            if x.shape[0] == 0:
                return x
            y = []
            last_index = 0
            assert isinstance(fwd_expert_count, np.ndarray)
            assert len(experts) == len(fwd_expert_count)
            for idx, expert_count in enumerate(fwd_expert_count):
                if expert_count <= 0:
                    continue
                y.append(
                    experts[idx](x[last_index : expert_count + last_index])
                )
                last_index = expert_count + last_index
            return paddle.concat(y, axis=0)

        if self.recompute_interval <= 0 or x.shape[0] == 0:
            x = experts_fwd(x, fwd_expert_count.numpy(), self.experts)
        else:
            x = recompute_hybrid(
                self.recompute_ctx,
                experts_fwd,
                x,
                fwd_expert_count.numpy(),
                self.experts,
            )

        out_batch_size = inp.shape[0]
        if len(gate.shape) == 2:
            out_batch_size *= gate.shape[1]

        x = MoEGather.apply(
            x,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            self.world_size,
            self.group,
        )

        x = x.reshape([-1, self.top_k, d_model])
        value = value.reshape([x.shape[0], 1, self.top_k])
        x = paddle.bmm(value, x).reshape([-1, d_model])

        if mp_size > 1:
            x = AllGather.apply(x, mp_rank, mp_size, self.mp_group)

        x = paddle.reshape_(x, origin_shape)

        return x

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients_with_group,
)
from paddle.nn import (
    Layer,
    functional as F,
)

####################################################
#                                                  #
#        Distributed Communication Operator        #
#                                                  #
####################################################


def scatter(input):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    rank = group.rank
    seq_len = input.shape[0]
    assert (
        seq_len % parallelism == 0
    ), "Input sequence length {} can't be divided exactly by sequence parallelism {}".format(
        seq_len, parallelism
    )
    interval = seq_len // parallelism
    input = paddle.slice(
        input, axes=[0], starts=[interval * rank], ends=[interval * (rank + 1)]
    )
    return input


def all_gather(input):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    output_shape = input.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    group.process_group.all_gather(input, output).wait()
    return output


def reduce_scatter(input):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), "Input sequence length {} can't be divided exactly by sequence parallelism {}".format(
        input.shape[0], parallelism
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(
        output, input, op=dist.ReduceOp.SUM, group=group, sync_op=True
    )
    return output


class ScatterOp(PyLayer):
    # input shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def forward(ctx, input):
        return scatter(input)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad)


class GatherOp(PyLayer):
    # input shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def forward(ctx, input):
        return all_gather(input)

    @staticmethod
    def backward(ctx, grad):
        return scatter(grad)


# All gather along the first dim during forward pass
# All reduce and scatter along the first dim during backward pass
class AllGatherOp(PyLayer):
    # input shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def forward(ctx, input):
        return all_gather(input)

    # grad shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def backward(ctx, grad):
        return reduce_scatter(grad)


# All reduce and scatter along the first dim during forward pass
# All gather along the first dim during backward pass
class ReduceScatterOp(PyLayer):
    # input shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def forward(ctx, input):
        return reduce_scatter(input)

    # grad shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad)


###################################################
#                                                 #
#        Modified Parallel Linear Operator        #
#                                                 #
###################################################


def mark_as_sequence_parallel_parameter(parameter):
    parameter.sequence_parallel = True


def is_sequence_parallel_parameter(parameter):
    return getattr(parameter, "sequence_parallel", False)


def create_fused_allreduce_gradient_hook(parameter_list, accumulation_steps):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()

    step = [0]
    accumulation_steps *= len(parameter_list)

    def __impl__(grad):
        step[0] += 1
        if step[0] == accumulation_steps:
            step[0] = 0
            fused_allreduce_gradients_with_group(
                parameter_list, group=group, scale=1.0
            )
        return grad

    return __impl__


def create_non_fused_allreduce_gradient_hook(param, accumulation_steps):
    hcg = fleet.get_hybrid_communicate_group()
    pg = hcg.get_model_parallel_group().process_group
    step = [0]

    @paddle.autograd.no_grad()
    def __impl__():
        step[0] += 1
        if (step[0] % accumulation_steps) == 0:
            if hasattr(param, "main_grad"):
                pg.allreduce(param.main_grad).wait()
            else:
                pg.allreduce(param.grad).wait()

    return __impl__


def register_sequence_parallel_allreduce_hooks(
    model, accumulation_steps, fuse_sequence_parallel_allreduce
):
    if accumulation_steps <= 0 or not paddle.distributed.is_initialized():
        return

    mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    if mp_group.nranks <= 1:
        return

    params = []
    for p in model.parameters():
        if is_sequence_parallel_parameter(p):
            params.append(p)

    if fuse_sequence_parallel_allreduce:
        hook = create_fused_allreduce_gradient_hook(params, accumulation_steps)
        for p in params:
            p._register_backward_hook(hook)
    else:
        for p in params:
            hook = create_non_fused_allreduce_gradient_hook(
                p, accumulation_steps
            )
            p._register_backward_hook(hook)


def is_fused_matmul_bias_supported():
    if (
        paddle.is_compiled_with_cuda()
        and not paddle.is_compiled_with_rocm()
        or paddle.is_compiled_with_xpu()
    ):
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


class ColumnSequenceParallelLinear(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=None,
        gather_output=True,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        hcg = fleet.get_hybrid_communicate_group()
        self.model_parallel_group = (
            hcg.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            hcg.get_model_parallel_group().nranks
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1

        assert (
            gather_output is False
        ), "If sequence_parallel is True, \
                                        gather_output is False"

        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            f"Number of column of the weight for linear ({out_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in ColumnSequenceParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher, or use xpu version."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear

    def forward(self, x):
        # sequence parallelism is same as model parallelism
        # if sequence parallel is true, input shape is [s, b, h]
        # else input shape is [b, s, h]
        if self.is_mp:
            input_parallel = AllGatherOp.apply(x)
        else:
            input_parallel = x
        output = self.linear(
            input_parallel, self.weight, self.bias, name=self._name
        )
        return output


class MPScale(PyLayer):
    @staticmethod
    def forward(ctx, x, mp_degree):
        out = paddle.scale(x, 1.0 / mp_degree)
        return out

    @staticmethod
    def backward(ctx, dout):
        return dout


class RowSequenceParallelLinear(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=True,
        input_is_parallel=False,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert (
            input_is_parallel is True
        ), "If sequence_parallel is True, \
                                           input_is_parallel should be true."

        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name

        hcg = fleet.get_hybrid_communicate_group()
        self.model_parallel_group = (
            hcg.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            hcg.get_model_parallel_group().nranks
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            hcg.get_model_parallel_group().rank
            if mp_group is None
            else mp_group.rank
        )

        self.is_mp = self.world_size > 1
        assert in_features % self.world_size == 0, (
            f"Number of row of the weight for linear ({in_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        # if sequence parallel is true,
        # register hook to all_reduce gradient of weight and bias
        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            if self.is_mp:
                mark_as_sequence_parallel_parameter(self.bias)
        else:
            self.bias = None

        self.linear = F.linear

        self.mp_scale = None
        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in RowParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear
            if self.is_mp and has_bias:
                self.mp_scale = MPScale.apply

    def forward(self, x):
        input_parallel = x
        if self.is_mp:
            if self.mp_scale is not None:
                bias = self.mp_scale(self.bias, self.world_size)
            else:
                bias = None
            output_parallel = self.linear(
                input_parallel, self.weight, bias, name=self._name
            )
            output_ = ReduceScatterOp.apply(output_parallel)
            # if self.bias is not none, sequence parallel will use
            # register_hook to all_reduce self.bias
            if bias is None and self.bias is not None:
                output = output_ + self.bias
            else:
                output = output_
        else:
            output = self.linear(
                input_parallel, self.weight, self.bias, name=self._name
            )
        return output

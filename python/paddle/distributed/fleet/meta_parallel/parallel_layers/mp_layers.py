#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.dygraph.layers import Layer
from .random import get_rng_state_tracker
from paddle.nn import functional as F
from paddle import framework
from ...base import topology as tp
from paddle.autograd import PyLayer
# from paddle.distributed import ReduceOp

__all__ = []

# Follow this paper to achieve the file:
# Shoeybi M, Patwary M, Puri R, et al. Megatron-lm: Training multi-billion parameter 
# language models using model parallelism[J]. arXiv preprint arXiv:1909.08053, 2019. (https://arxiv.org/abs/1909.08053)


class VocabParallelEmbedding(Layer):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 weight_attr=None,
                 name=None):
        super(VocabParallelEmbedding, self).__init__()

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        )
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()

        self.origin_num_embeddings = num_embeddings
        self.is_mp = (self.world_size > 1)

        assert num_embeddings % self.world_size == 0, (
            "The length of the vocabulary must be divisible by the parallelism degree of MP"
        )

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name

        if self.is_mp:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    attr=self._weight_attr,
                    shape=self._size,
                    dtype=self._dtype,
                    is_bias=False)
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=self._size,
                dtype=self._dtype,
                is_bias=False)

        self.weight.is_distributed = True

    def forward(self, x):
        if self.is_mp:
            output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)
            output = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True)
        else:
            output = F.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name)
        return output


class ColumnParallelLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=None,
                 gather_output=True,
                 name=None):
        super(ColumnParallelLinear, self).__init__()

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        )
        self._name = name
        self.is_mp = (self.world_size > 1)

        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(out_features,
                                                            self.world_size))
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False)
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)

        self.weight.is_distributed = True

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True)
            self.bias.is_distributed = True
        else:
            self.bias = None

    def forward(self, x):
        # use inner api to process identity
        if self.is_mp:
            input_parallel = paddle.distributed.collective._c_identity(
                x, group=self.model_parallel_group)
        else:
            input_parallel = x

        output_parallel = F.linear(
            input_parallel, self.weight, self.bias, name=self._name)

        if self.gather_output and self.is_mp:
            output = paddle.distributed.collective._c_concat(
                output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output


class RowParallelLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=True,
                 input_is_parallel=False,
                 name=None):
        super(RowParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        )
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()

        self.is_mp = (self.world_size > 1)
        assert in_features % self.world_size == 0, (
            "Number of row of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(in_features,
                                                            self.world_size))

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False)
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)

        self.weight.is_distributed = True

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True)
        else:
            self.bias = None

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = paddle.distributed.collective._c_split(
                x, group=self.model_parallel_group)

        output_parallel = F.linear(input_parallel, self.weight, name=self._name)

        if self.is_mp:
            output_ = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True)
        else:
            output_ = output_parallel

        output = output_ + self.bias if self.bias is not None else output_
        return output


class _VocabParallelCrossEntropy(PyLayer):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
        rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()

        # Maximum value along vocab dimension across all GPUs.
        logits_max = paddle.max(x=vocab_parallel_logits, axis=-1)[0]
        paddle.distributed.collective.all_reduce(
            logits_max,
            op=paddle.distributed.ReduceOp.MAX,
            group=model_parallel_group)

        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(
            axis=-1)

        partition_vocab_size = vocab_parallel_logits.shape()[-1]
        vocab_start_index = rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        target_mask = paddle.logical_or((target < vocab_start_index),
                                        (target >= vocab_end_index))
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
        masked_target_1d = masked_target.reshape(-1)
        arange_1d = paddle.arange(0, logits_2d.size()[0], 'int32')

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # All reduce is needed to get the chunks from other GPUs.
        paddle.distributed.collective.all_reduce(
            predicted_logits,
            op=paddle.distributed.ReduceOp.SUM,
            group=model_parallel_group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        # exp_logits = vocab_parallel_logits
        exp_logits = paddle.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        paddle.distributed.collective.all_reduce(
            sum_exp_logits,
            op=paddle.distributed.ReduceOp.SUM,
            group=model_parallel_group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = paddle.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits = exp_logits / (sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.shape()[-1]
        grad_2d = grad_input.reshape(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = paddle.arange(
            start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - target_mask.reshape(-1))

        # Finally elementwise multiplication with the output gradients.
        grad_input = paddle.mul(grad_input, grad_output.unsqueeze(dim=-1))

        return grad_input, None


class ParallelCrossEntropy(Layer):
    def __init__(self, name=None):
        super(ParallelCrossEntropy, self).__init__()
        # self.weight = weight
        # self.reduction = reduction
        # self.ignore_index = ignore_index
        # self.soft_label = soft_label
        # self.axis = axis
        # self.use_softmax = use_softmax
        self.name = name
        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        )
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()

    def forward(self, input, label):
        loss = paddle.distributed.collective._c_softmax_with_cross_entropy(
            input, label, group=self.model_parallel_group)

        # loss = _VocabParallelCrossEntropy.apply(input, label)
        # ret = paddle.nn.functional.cross_entropy(
        #     input,
        #     label,
        #     weight=self.weight,
        #     ignore_index=self.ignore_index,
        #     reduction=self.reduction,
        #     soft_label=self.soft_label,
        #     axis=self.axis,
        #     use_softmax=self.use_softmax,
        #     name=self.name)

        return loss

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

        per_part_size = (
            num_embeddings + self.world_size - 1) // self.world_size
        last_part_size = num_embeddings - per_part_size * (self.world_size - 1)
        if self.rank == self.world_size - 1:
            per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index
        self.per_part_size = per_part_size

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
            self.weight[per_part_size - 1] = 0.0
            self.weight.is_distributed = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_embeddings, embedding_dim],
                dtype=self._dtype,
                is_bias=False)

    def forward(self, x):
        if not self.is_mp:
            return F.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name)

        origin_input_shape = x.shape
        if len(origin_input_shape) == 2:
            x = paddle.unsqueeze(x, axis=-1)
        else:
            assert origin_input_shape[-1] == 1, (
                "The last dimension size of x must be 1.")
        x_shard = paddle.shard_index(x, self.origin_num_embeddings,
                                     self.world_size, self.rank,
                                     self.per_part_size - 1)
        if len(origin_input_shape) == 2:
            x_shard = paddle.squeeze(x_shard, axis=-1)

        emb_out = F.embedding(
            x_shard,
            weight=self.weight,
            padding_idx=self.per_part_size - 1,
            sparse=False,
            name=self._name)

        emb_out = paddle.distributed.collective._mp_allreduce(
            emb_out,
            group=self.model_parallel_group,
            use_calc_stream=True,
            use_model_parallel=True)
        return emb_out


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
                output_parallel,
                nranks=self.world_size,
                group=self.model_parallel_group)
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
                x,
                rank=self.rank,
                nranks=self.world_size,
                group=self.model_parallel_group)

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

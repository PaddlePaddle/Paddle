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
from .layers_help import identity_in_model_parallel, gather_in_model_parallel, reduce_in_model_parallel, scatter_in_model_parallel

__all__ = [
    'VocabParallelEmbedding', 'ColumnParallelLinear', 'RowParallelLinear'
]

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

        per_part_size = (
            num_embeddings + self.world_size - 1) // self.world_size
        last_part_size = num_embeddings - per_part_size * (self.world_size - 1)
        if self.rank == self.world_size - 1:
            per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index
        self.per_part_size = per_part_size

        self.embedding = paddle.nn.Embedding(
            per_part_size,
            embedding_dim,
            padding_idx=per_part_size - 1,
            sparse=False,
            weight_attr=weight_attr,
            name=name)
        self.embedding.weight.is_distributed = True

    def forward(self, x):
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

        emb_out_ = self.embedding(x_shard)
        emb_out = reduce_in_model_parallel(emb_out_)
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

        self.name = name
        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(out_features,
                                                            self.world_size))
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
            shape=[in_features, self.output_size_per_partition],
            attr=self._weight_attr,
            dtype=self._dtype)
        self.weight.is_distributed = True

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype)
            self.bias.is_distributed = True
        else:
            self.bias = None

    def forward(self, x):
        input_parallel = identity_in_model_parallel(x)
        output_parallel = F.linear(
            input_parallel, self.weight, self.bias, name=self.name)
        if self.gather_output:
            output = gather_in_model_parallel(output_parallel)
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
        self.name = name

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        )
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        )
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()

        assert in_features % self.world_size == 0, (
            "Number of row of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(in_features,
                                                            self.world_size))

        self.input_size_per_partition = in_features // self.world_size

        self.weight = self.create_parameter(
            shape=[self.input_size_per_partition, self.out_features],
            attr=self._weight_attr,
            dtype=self._dtype)
        self.weight.is_distributed = True

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype)
        else:
            self.bias = None

    def forward(self, x):
        if self.input_is_parallel:
            input_parallel = x
        else:
            # split last dim
            input_parallel = scatter_in_model_parallel(x)

        output_parallel = F.linear(input_parallel, self.weight, name=self.name)
        output_ = reduce_in_model_parallel(output_parallel)
        output = output_ + self.bias if self.bias is not None else output_
        return output

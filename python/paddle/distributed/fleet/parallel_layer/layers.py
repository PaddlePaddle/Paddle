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
from paddle.distributed.fleet import fleet

__all__ = ['ParallelLinear', 'ParallelEmbedding']


class ParallelLinear(Layer):
    def __init__(self,
                 size,
                 axis=0,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super(ParallelLinear, self).__init__()

        self.hcg = fleet.get_hybrid_communicate_group()
        self.model_parallel_group = self.hcg.get_model_parallel_group()

        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            per_part_size = size[0] // num_partitions
            linear_size = (per_part_size, size[1])

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            per_part_size = size[1] // num_partitions
            linear_size = (size[0], per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        self.num_partitions = num_partitions
        self.linear = paddle.nn.Linear(
            linear_size[0],
            linear_size[1],
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)
        self.linear_size = linear_size

        self.axis = axis
        self.gather_out = gather_out

    def forward(self, x):
        linear_out = self.linear(x)
        if self.gather_out:
            if self.axis == 0:
                paddle.distributed.all_reduce(
                    linear_out, group=self.model_parallel_group)
            else:
                output = []
                paddle.distributed.all_gather(
                    output, linear_out, group=self.model_parallel_group)
                linear_out = paddle.concat(
                    output, axis=len(linear_out.shape) - 1)
        return linear_out


# class ParallelEmbedding(Layer):
#     def __init__(self,
#                  size,
#                  num_partitions=1,
#                  param_attr=None,
#                  bias_attr=None,
#                  name=None):
#         super(ParallelEmbedding, self).__init__()
#         assert fleet._role_maker, ("To use paddle.distributed.split, "
#                                    "you must call fleet.init() firstly.")
#         rank = fleet.worker_index()
#         nranks = fleet.worker_num()
#         # rank within a model parallel group

#         inner_rank = rank % num_partitions
#         self.inner_rank = inner_rank
#         per_part_size = (size[0] + num_partitions - 1) // num_partitions
#         last_part_size = size[0] - per_part_size * (num_partitions - 1)
#         if inner_rank == num_partitions - 1: 
#             per_part_size = last_part_size
#         per_part_size += 1  # make the last row as the padding index
#         self.per_part_size = per_part_size
#         self.origin_num_embeddings = size[0]
#         self.embedding = paddle.nn.Embedding(
#             per_part_size,
#             size[1],
#             padding_idx=per_part_size - 1,
#             sparse=False,
#             weight_attr=param_attr,
#             name=name)
#         self.embedding.weight.is_distributed = True
#         self.num_partitions = num_partitions

#     def forward(self,x):
#         origin_input_shape = x.shape
#         if len(origin_input_shape) == 2:
#             x = paddle.unsqueeze(x, axis=-1)
#         else:
#             assert origin_input_shape[-1] == 1, (
#                 "The last dimension size of x must be 1.")
#         x_shard = paddle.shard_index(x, self.origin_num_embeddings, self.num_partitions,
#                                      self.inner_rank, self.per_part_size - 1)
#         if len(origin_input_shape) == 2:
#             x_shard = paddle.squeeze(x_shard, axis=-1)
#         emb_out = self.embedding(x_shard)
#         paddle.distributed.all_reduce(emb_out, group=0)
#         return emb_out

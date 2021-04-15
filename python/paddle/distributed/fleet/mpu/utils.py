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

from paddle.autograd import PyLayer
from ..base import topology as tp
import paddle


def _reduce(x):
    if tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() == 1:
        return x

    paddle.distributed.all_reduce(
        x, group=tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group())

    return x


def _split(x):
    world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()

    if world_size == 1:
        return x

    rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
    last_dim = len(x.shape) - 1
    input_list = paddle.split(x, num_or_sections=world_size, axis=last_dim)
    output = input_list[rank]

    return output


def _gather(x):
    world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()

    if world_size == 1:
        return x

    output = []
    paddle.distributed.all_gather(
        output, x, group=tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group())

    output = paddle.concat(output, axis=len(x.shape) - 1)

    return output


class _IdentityInModelParallel(PyLayer):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, dx):
        return _reduce(dx)


class _ReduceInModelParallel(PyLayer):
    @staticmethod
    def forward(ctx, x):
        return _reduce(x)

    @staticmethod
    def backward(ctx, dx):
        return dx


class _ScatterInModelParallel(PyLayer):
    @staticmethod
    def forward(ctx, x):
        return _split(x)

    @staticmethod
    def backward(ctx, dx):
        return _gather(dx)


class _GatherInModelParallel(PyLayer):
    @staticmethod
    def forward(ctx, x):
        return _gather(x)

    @staticmethod
    def backward(ctx, dx):
        return _split(dx)


def identity_in_model_parallel(input_):
    return _IdentityInModelParallel.apply(input_)


def reduce_in_model_parallel(input_):
    return _ReduceInModelParallel.apply(input_)


def scatter_in_model_parallel(input_):
    return _ScatterInModelParallel.apply(input_)


def gather_in_model_parallel(input_):
    return _GatherInModelParallel.apply(input_)

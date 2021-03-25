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
import collections
import numpy as np
from itertools import product
from functools import reduce
__all__ = ['CommunicateTopology']


class CommunicateTopology(object):
    def __init__(self, hybrid_names, dims):
        self._parallel_names = hybrid_names
        self._dims = dims
        self.coordinate = collections.namedtuple('Coordinate',
                                                 self._parallel_names)
        self._word_size = reduce(lambda x, y: x * y, self._dims)

        ranges = [range(d) for d in self._dims]
        all_coordinate = [self.coordinate(*x) for x in product(*ranges)]

        self._coord2rank = dict(zip(all_coordinate, range(len(all_coordinate))))
        self._rank2coord = dict(
            zip(self._coord2rank.values(), self._coord2rank.keys()))

    def get_parallel_names(self):
        return self._parallel_names

    def get_dims(self):
        return self._dims

    def word_size(self):
        return self._word_size

    def get_rank(self, **args):
        assert len(args) == len(self._dims)
        key = self.coordinate(**args)
        assert self._coord2rank.has_key(key)
        return self._coord2rank[key]

    def get_coord(self, rank):
        assert rank < self._word_size
        assert self._rank2coord.has_key(rank)
        return self._rank2coord[rank]

    def get_axis_list(self, axis_name, index):
        axis = self._parallel_names.index(axis_name)
        ranks = [
            self._coord2rank[coord] for coord in self._coord2rank.keys()
            if coord[axis] == index
        ]
        ranks.sort()
        return ranks

    def get_dim_num(self, axis_name):
        assert axis_name in self._parallel_names
        return self._dims[self._parallel_names.index(axis_name)]

    def get_comm_list(self, axis_name):
        assert axis_name in self._parallel_names
        other_axis_names = [
            name for name in self._parallel_names if name != axis_name
        ]

        ranges = []
        for name in other_axis_names:
            dim_num = self.get_dim_num(name)
            ranges.append(range(dim_num))

        all_result = []
        for x in product(*ranges):
            key_coord = {}
            for other_name in other_axis_names:
                key_coord[other_name] = x[other_axis_names.index(other_name)]

            result = []
            for i in range(0, self.get_dim_num(axis_name)):
                key_coord[axis_name] = i
                result.append(self._coord2rank[self.coordinate(**key_coord)])
            all_result.append(result)

        return all_result


class HybridCommunicateGroup(object):
    def __init__(self, topology):
        self.nranks = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self._topo = topology

        self._num_data_parallel = self._topo.get_dim('data')
        self._num_model_parallel = self._topo.get_dim('model')
        self._num_pipe_parallel = self._topo.get_dim('pipe')

        assert self._check_vaild_topo(
        ), "Here is an unreasonable topogy setting"

        # create comm group for data parallel
        self.dp_group, self.dp_comm_group = self._set_comm_group("data")
        print("data parallel group", self.dp_group)

        # create comm group for model parallel
        self.mp_group, self.mp_comm_group = self._set_comm_group("model")
        print("model parallel group", self.mp_group)

    def _check_vaild_topo(self):
        return self._num_data_parallel * self._num_model_parallel * self._num_pipe_parallel == self.nranks

    def _get_data_parallel_id(self):
        return self._topo.get_coord(self.global_rank).data

    def _set_comm_group(self, parallel_method="data"):
        parallel_group = []
        parallel_comm_group = None
        parallel_groups = self._topo.get_comm_list(parallel_method)

        for g in parallel_groups:
            comm_group = dist.new_group(ranks=group)
            if global_rank in group:
                parallel_group = group
                parallel_comm_group = comm_group

        assert len(parallel_group) > 0
        assert parallel_comm_group is not None

        return parallel_group, parallel_comm_group

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
# limitations under the License

import paddle
import paddle.fluid.core as core
from ..collective import _get_global_env
from ..collective import _new_ring_id
from ...fluid.framework import in_dygraph_mode
from ...fluid.layers.tensor import fill_constant


def get_all_process_groups():
    global _g_process_group_map
    return _g_process_group_map.values()


def get_process_group(group_id, g_process_group_map=None):
    global _g_process_group_map
    return _g_process_group_map.get(
        group_id,
        None) if g_process_group_map is None else g_process_group_map.get(
            group_id, None)


def get_world_process_group():
    global _g_process_group_map
    return _g_process_group_map[0]


def new_process_group(ranks):
    global _g_process_group_map
    # A key constructed from ranks is used for avoiding duplication 
    new_key = ''.join(map(str, sorted(ranks)))
    for pg_id, pg in _g_process_group_map.items():
        cur_key = ''.join(map(str, sorted(pg.ranks)))
        if pg_id != 0 and new_key == cur_key:
            return pg
    # If not matching the existing one, construt a new process group
    num_groups = len(_g_process_group_map)
    # Note: our process group may interfere with the original implementation
    # so the created group id should start from the original _new_ring_id()
    group_id = _new_ring_id() + num_groups + 1
    new_pg = ProcessGroup(group_id, ranks)
    _g_process_group_map[group_id] = new_pg
    return new_pg


# This implementation refers to lots of Paddle/python/paddle/distributed/collective.py,
# Fleet also has a collective helper which uses ops to initialize communication in 
# Paddle/python/paddle/distributed/fleet/meta_optimizers/common.py. We use the first one
# because it seems simple. This should be enhanced to manage the process membership and 
# the instantiation process in a more general way. In the future, the process group may 
# handle the communication implementation choice.
class ProcessGroup:
    def __init__(self, group_id, ranks):
        if group_id == 0 and get_process_group(0) is not None:
            assert group_id != 0, "Process group id 0 is reserved for all ranks."
        self._group_id = group_id
        self._ranks = sorted(ranks)
        # Add the current ranks into group 0
        if group_id != 0:
            global _g_process_group_map
            _g_process_group_map[0].add_ranks(ranks)
        self._is_instantiate = False

    @property
    def id(self):
        return self._group_id

    @property
    def ranks(self):
        return self._ranks

    @property
    def nranks(self):
        return len(self._ranks)

    def add_ranks(self, new_ranks):
        if set(new_ranks) <= set(self.ranks):
            return
        else:
            assert self.is_instantiate() == False, \
                "Cannot add new ranks after instantiating the process group"
        self._ranks.extend(new_ranks)
        self._ranks = sorted(list(set(self.ranks)))

    def local_rank(self, global_rank):
        if global_rank in self.ranks:
            return self.ranks.index(global_rank)
        else:
            assert False, \
                "Rank {} doesn't belong to this group".format(global_rank)

    def is_instantiate(self):
        return self._is_instantiate

    def instantiate(self):
        if self._is_instantiate:
            return
        ring_id = self.id
        genv = _get_global_env()
        global_rank = genv.rank

        if self.nranks >= 2:
            strategy = core.ParallelStrategy()
            strategy.nranks = self.nranks
            strategy.local_rank = self.local_rank(global_rank)
            strategy.trainer_endpoints = [
                genv.trainer_endpoints[i] for i in self.ranks
            ]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                core.NCCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            else:
                assert False, ("No CUDA device found")

        # TODO(shenliang03): This is a temporary solution to solve the problem of 
        # hang caused by cross-creation of new_group
        tmp = paddle.to_tensor(
            [1], dtype="int32") if in_dygraph_mode() else fill_constant(
                [0], dtype="int32", value="1")
        paddle.distributed.all_reduce(tmp, use_calc_stream=True)
        paddle.distributed.wait(tmp)

        self._is_instantiate = True

    # def __eq__(self, other):
    #     if not isinstance(other, ProcessGroup):
    #         return False
    #     if self.id != other.id:
    #         return False
    #     return True

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    def __str__(self):
        string = "id: {}, nranks: {}, ranks: {}.".format(
            self.id, self.nranks, ", ".join(map(str, self.ranks)))
        return string


# Note that Process group 0 is reserved for representing all ranks.
# At the begining, group 0 is empty and new ranks will be added automatically. 
_g_process_group_map = {}
_g_process_group_map[0] = ProcessGroup(0, [])

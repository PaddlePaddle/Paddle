# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class Group():
    """
    The abstract representation of group.
    """

    def __init__(self, group_rank, id, ranks, pg=None, name=None):
        self._group_rank = group_rank
        self._world_size = len(ranks) if group_rank >= 0 else -1
        self._id = id
        self._ranks = ranks
        self._pg = pg
        self._name = name

    @property
    def rank(self):
        return self._group_rank

    @property
    def ranks(self):
        return self._ranks

    @property
    def nranks(self):
        return len(self._ranks)

    @property
    def name(self):
        return self._name

    @property
    def process_group(self):
        return self._pg

    @property
    def world_size(self):
        return self._world_size

    @property
    def id(self):
        return self._id

    def is_member(self):
        if self.rank < 0:
            return False
        if self.nranks < 2:
            return False
        return True

    def get_group_rank(self, rank):
        if self.is_member():
            return self.ranks.index(rank)
        else:
            return -1

    def __repr__(self):
        debug_str = "rank: {}, nranks: {}, id: {}, ranks: ".format(
            self.rank, self.nranks, self.id)
        debug_str += ", ".join(map(str, self.ranks))
        debug_str += "; name: "
        debug_str += self.name if self.name else "None"
        return debug_str

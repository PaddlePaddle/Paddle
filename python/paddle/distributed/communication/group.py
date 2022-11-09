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

import warnings
import paddle.distributed as dist


class Group:
    """
    The abstract representation of group.
    """

    def __init__(self, rank_in_group, id, ranks, pg=None, name=None):
        self._rank_in_group = rank_in_group
        self._world_size = len(ranks) if rank_in_group >= 0 else -1
        self._id = id
        self._ranks = ranks
        self._pg = pg
        self._name = name

    @property
    def rank(self):
        return self._rank_in_group

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
    def backend(self):
        return self._pg.name()

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
            self.rank, self.nranks, self.id
        )
        debug_str += ", ".join(map(str, self.ranks))
        debug_str += "; name: "
        debug_str += self.name if self.name else "None"
        return debug_str


class _GroupManager:
    global_group_id = 0
    group_map_by_id = {}


def _get_global_group():
    if _GroupManager.global_group_id not in _GroupManager.group_map_by_id:
        raise RuntimeError("The global group is not initialized.")
    return _GroupManager.group_map_by_id[_GroupManager.global_group_id]


def _add_new_group(group):
    if group.id in _GroupManager.group_map_by_id:
        raise RuntimeError(
            "The group with id {} already exist.".format(group.id)
        )
    _GroupManager.group_map_by_id[group.id] = group


def _is_global_group(group):
    return group.id == _GroupManager.global_group_id


def _warn_cur_rank_not_in_group(group):
    global_rank = dist.get_rank()
    if group and not group.is_member():
        warnings.warn(
            "Current global rank {} is not in group {}".format(
                global_rank, group.name
            )
        )
        return True
    return False


def _get_or_throw_group_rank(global_rank, group):
    group_rank = group.get_group_rank(global_rank)
    assert (
        group_rank >= 0
    ), "The input rank {} can not be found inside the group {}".format(
        global_rank, group.name
    )
    return group_rank


def is_initialized():
    """

    Check whether the distributed environment has been initialized

    Returns:
        `True` if distributed environment has been initialized, otherwise `False`.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle

            print(paddle.distributed.is_initialized())
            # False

            paddle.distributed.init_parallel_env()
            print(paddle.distributed.is_initialized())
            # True

    """
    return _GroupManager.global_group_id in _GroupManager.group_map_by_id


def destroy_process_group(group=None):
    """
    Destroy a given group for communication

    Args:
        group (Group, optional): The group to be destroyed. All of process groups, including
                                        the default group, will be destroyed and the distributed
                                        environment will be deinitialized.

    Returns : None

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            group = dist.new_group([0, 1])

            dist.destroy_process_group(group)
            print(dist.is_initialized())
            # True
            dist.destroy_process_group()
            print(dist.is_initialized())
            # False

    """
    group = _get_global_group() if group is None else group
    assert (
        group.id in _GroupManager.group_map_by_id
    ), "Destroy group with id {} is invalid.".format(group.id)
    if _is_global_group(group):
        _GroupManager.group_map_by_id.clear()
    else:
        del _GroupManager.group_map_by_id[group.id]


def get_group(id=0):
    """

    Get group instance by group id.

    Args:
        id (int): the group id. Default value is 0.

    Returns:
        Group: the group instance.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            gid = paddle.distributed.new_group([2,4,6])
            paddle.distributed.get_group(gid.id)

    """

    if id in _GroupManager.group_map_by_id:
        return _GroupManager.group_map_by_id[id]
    warnings.warn("Group {} is not initialized.".format(id))
    return None

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

import paddle.distributed as dist


class StrategyGroupBase:
    """
    The base class of communication group with distributed strategy.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of strategy group.

    Examples:
        .. code-block:: python

            import paddle.distributed as dist
            from paddle.distributed.fleet.base.strategy_group import StrategyGroupBase

            dist.init_parallel_env()
            strategy_group = dist.fleet.base.strategy_group.StrategyGroupBase([[0, 1], [2, 3]])
            print(strategy_group.world_size)  # 2

    """

    def __init__(self, list_of_ranks):
        assert (
            dist.is_initialized()
        ), "The global communication group need to be initialized."
        assert len(list_of_ranks), "The list_of_ranks can not be empty."
        self._rank = dist.get_rank()
        self._list_of_ranks = list_of_ranks
        self._group = self._create_group()

    @property
    def world_size(self):
        """
        The world size of communication group.

        Returns:
            Integer if the world_size of each group are equal, or a list of world_size if they are not equal.
        """
        world_size_list = []
        for ranks in self._list_of_ranks:
            world_size_list.append(len(ranks))
        is_value = all(
            world_size == world_size_list[0] for world_size in world_size_list
        )
        return world_size_list[0] if is_value else world_size_list

    @property
    def group(self):
        """
        The communication group which current rank belongs to.

        Returns:
            Group if current rank only belong to single communication group, or a list of Group if it belongs many.
        """
        return self._group

    def _create_group(self):
        list_of_group = []
        for ranks in self._list_of_ranks:
            group = dist.new_group(ranks=ranks)
            if self._rank in ranks:
                list_of_group.append(group)
        assert (
            len(list_of_group) > 0
        ), "Rank {} does not belong to the list_of_ranks {}.".format(
            self._rank, self._list_of_ranks
        )
        return list_of_group if len(list_of_group) > 1 else list_of_group[0]


class DPGroup(StrategyGroupBase):
    """
    The communication group strategy for data parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of data parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        super().__init__(list_of_ranks)
        assert not isinstance(
            self.group, list
        ), f"Rank {self._rank} belongs to multi dp groups"


class MPGroup(StrategyGroupBase):
    """
    The communication group strategy for model parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of model parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        super().__init__(list_of_ranks)
        assert not isinstance(
            self.group, list
        ), f"Rank {self._rank} belongs to multi mp groups"


class ShardingGroup(StrategyGroupBase):
    """
    The communication group strategy for sharding parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of sharding parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        super().__init__(list_of_ranks)
        assert not isinstance(
            self.group, list
        ), f"Rank {self._rank} belongs to multi sharding groups"


class PPGroup(StrategyGroupBase):
    """
    The communication group strategy for pipeline parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of pipeline parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        super().__init__(list_of_ranks)
        assert not isinstance(
            self.group, list
        ), f"Rank {self._rank} belongs to multi pp groups"

        self._send_next_group = None
        self._send_prev_group = None
        self._recv_next_group = None
        self._recv_prev_group = None
        self._rank_of_next_stage = None
        self._rank_of_prev_stage = None

        if self.world_size > 1:
            self._create_p2p_group()

    @property
    def rank_of_prev_stage(self):
        """
        Rank of the previous pp stage.

        Returns:
            The global rank of previous pp stage. `None` if without previous.
        """
        return self._rank_of_prev_stage

    @property
    def rank_of_next_stage(self):
        """
        Rank of the next pp stage.

        Returns:
            The global rank of next pp stage. `None` if without next.
        """
        return self._rank_of_next_stage

    @property
    def p2p_groups(self):
        """
        Communication subgroup in order to switch data with previous and next stage.

        Returns:
            Four subgroups including send/recv to/from prev/next.
        """
        return (
            self._send_next_group,
            self._send_prev_group,
            self._recv_next_group,
            self._recv_prev_group,
        )

    def _create_p2p_group(self):
        degree = self.world_size
        for ranks in self._list_of_ranks:
            for idx, rank in enumerate(ranks):
                next_rank = ranks[(idx + 1) % degree]
                prev_rank = ranks[(idx - 1) % degree]

                if self._rank == rank:
                    self._rank_of_next_stage = next_rank
                    self._rank_of_prev_stage = prev_rank

                next_group = dist.new_group(ranks=[rank, next_rank])

                if self._rank == rank:
                    self._send_next_group = next_group
                elif self._rank == next_rank:
                    self._recv_prev_group = next_group

                prev_group = dist.new_group(ranks=[prev_rank, rank])
                if self._rank == rank:
                    self._send_prev_group = prev_group
                elif self._rank == prev_rank:
                    self._recv_next_group = prev_group

        assert (
            self._send_next_group
            and self._send_prev_group
            and self._recv_next_group
            and self._recv_prev_group
        ), "Error occurs while creating p2p group for rank {}.".format(
            self._rank
        )

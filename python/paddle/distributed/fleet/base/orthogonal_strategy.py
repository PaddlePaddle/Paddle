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

import collections
import functools
import itertools

import paddle.distributed as dist
from paddle.distributed.fleet.base.strategy_group import StrategyGroupBase


class OrthogonalStrategy:
    """
    A hybrid of multiple distributed strategies. Strategies need to be orthogonal, means the ranks are organized like
    a square if there are two strategies, a cube if there are three strategies, etc.

    Args:
        list_of_strategy(list): Strategy in the list should be represented as tuple, format as (strategy_name, degree, strategy_class).
        fused_strategy_dict(dict, optional): Exist strategies can be fused to new strategy. Use the name of new strategy as key, a list of
            strategy names you want to fuse as value.

    Returns:
        The instance of strategy.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.distributed.fleet.base.strategy_group import DPGroup, MPGroup, PPGroup
            >>> from paddle.distributed.fleet.base.orthogonal_strategy import OrthogonalStrategy

            >>> dist.init_parallel_env()
            >>> strategy = OrthogonalStrategy([("dp", 2, DPGroup), ("mp", 2, MPGroup), ("pp", 2, PPGroup)], fused_strategy_dict={"check": ["mp", "pp"]})

    """

    def __init__(
        self, list_of_strategy, fused_strategy_dict={}, strategy_rank_list=None
    ):
        self._list_of_strategy = list_of_strategy
        self._fused_strategy_dict = fused_strategy_dict
        self._strategy_rank_list = (
            strategy_rank_list
            if strategy_rank_list is not None
            else list(range(dist.get_world_size()))
        )
        self._name_to_group_dict = {}
        self._name_to_degree_dict = {}
        self._list_of_strategy_name = [
            strategy[0] for strategy in list_of_strategy
        ]
        self._list_of_degree = [strategy[1] for strategy in list_of_strategy]
        self._coordinate = collections.namedtuple(
            'Coordinate', self._list_of_strategy_name
        )
        self._check_valid_strategy()

        ranges = [range(degree) for degree in self._list_of_degree]
        list_of_coord = [
            self._coordinate(*coord) for coord in itertools.product(*ranges)
        ]

        self._coord_to_rank_dict = dict(
            zip(list_of_coord, self._strategy_rank_list)
        )

        for idx, strategy in enumerate(list_of_strategy):
            strategy_name = strategy[0]
            self._name_to_degree_dict[strategy_name] = strategy[1]
            rank_list = self._calc_rank_list(idx)
            self._name_to_group_dict[strategy_name] = strategy[2](
                rank_list,
            )

        self._name_to_fused_group_dict = {}
        self._create_fused_group()

    def strategy_group(self, name):
        """
        Get strategy group with specific name.

        Args:
            name: The name of strategy group

        Returns:
            An instance of specific strategy group.
        """
        assert (
            name in self._list_of_strategy_name
        ), f"Strategy group {name} is not created."
        return self._name_to_group_dict[name]

    def fused_strategy_group(self, name):
        """
        Get fused strategy group with specific name.

        Args:
            name: The name of fused strategy group

        Returns:
            (StrategyGroupBase): An instance of strategy group.
        """
        assert (
            name in self._name_to_fused_group_dict
        ), f"Fused strategy group {name} is not created."
        return self._name_to_fused_group_dict[name]

    def rank_in_strategy(self, name):
        """
        Get local rank in strategy group with specific name.

        Args:
            name: The name of strategy group

        Returns:
            (Integer): Local rank in specific strategy.
        """
        assert (
            name in self._list_of_strategy_name
        ), f"Strategy group {name} is not created."
        return self._name_to_group_dict[name].group.rank

    def _check_valid_strategy(self):
        assert len(self._list_of_strategy_name) == len(
            set(self._list_of_strategy_name)
        ), f"Defined duplicated strategies: {self._list_of_strategy_name}"
        num_of_ranks = functools.reduce(
            lambda x, y: x * y, self._list_of_degree
        )

        assert num_of_ranks == len(
            self._strategy_rank_list
        ), f"There are total {len(self._strategy_rank_list)} ranks, but need {num_of_ranks} ranks in this strategy."

        for fused_strategy in self._fused_strategy_dict.values():
            for strategy in fused_strategy:
                assert (
                    strategy in self._list_of_strategy_name
                ), f"Can not fuse strategy {strategy} without defined previous."

    def _create_fused_group(self):
        for name in self._fused_strategy_dict:
            fused_strategy = self._fused_strategy_dict[name]
            non_fused_strategy = list(
                set(self._list_of_strategy_name).difference(fused_strategy)
            )
            non_fused_ranges = []
            for strategy in non_fused_strategy:
                non_fused_ranges.append(
                    range(self._name_to_degree_dict[strategy])
                )
            fused_ranges = []
            for strategy in fused_strategy:
                fused_ranges.append(range(self._name_to_degree_dict[strategy]))

            rank_list = []
            for non_fused_ranks in itertools.product(*non_fused_ranges):
                coord_dict = {}
                ranks = []
                for i, non_fused_rank in enumerate(non_fused_ranks):
                    coord_dict[non_fused_strategy[i]] = non_fused_rank
                for fused_ranks in itertools.product(*fused_ranges):
                    for i, fused_rank in enumerate(fused_ranks):
                        coord_dict[fused_strategy[i]] = fused_rank
                    ranks.append(
                        self._coord_to_rank_dict[self._coordinate(**coord_dict)]
                    )
                rank_list.append(ranks)
            self._name_to_fused_group_dict[name] = StrategyGroupBase(rank_list)

    def _calc_rank_list(self, strategy_axis):
        ranges = []
        for idx, degree in enumerate(self._list_of_degree):
            if idx == strategy_axis:
                continue
            ranges.append(range(degree))

        rank_list = []
        for coord in itertools.product(*ranges):
            ranks = []
            for val in range(self._list_of_degree[strategy_axis]):
                coord_list = list(coord)
                coord_list.insert(strategy_axis, val)
                ranks.append(
                    self._coord_to_rank_dict[self._coordinate(*coord_list)]
                )
            rank_list.append(ranks)

        return rank_list

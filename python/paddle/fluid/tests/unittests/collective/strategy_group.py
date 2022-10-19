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

import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.base.strategy_group import StrategyGroupBase, DPGroup, MPGroup, PPGroup, ShardingGroup


def _check_using_all_reduce(group):
    data = paddle.to_tensor([1, 2, 3])
    result = paddle.to_tensor([2, 4, 6])
    dist.all_reduce(data, group=group)
    assert np.array_equal(data, result)


def _check_using_send(group, dst):
    data = paddle.to_tensor([1, 2, 3])
    dist.send(data, dst=dst, group=group)


def _check_using_recv(group, src):
    result = paddle.to_tensor([1, 2, 3])
    data = paddle.to_tensor([0, 0, 0])
    dist.recv(data, src=src, group=group)
    assert np.array_equal(data, result)


class TestStrategyGroupAPI(unittest.TestCase):

    def setUp(self):
        self._num_of_ranks = 2
        self._list_of_rank = [[0, 1]]
        self._list_of_ranks = [[0, 1], [0, 1]]
        dist.init_parallel_env()
        self._global_rank = dist.get_rank()
        self._peer_rank = 0 if self._global_rank == 1 else 1

    def test_strategy_group_base(self):
        strategy_group = StrategyGroupBase(self._list_of_rank)
        self.assertEqual(strategy_group.world_size, self._num_of_ranks)
        self.assertEqual(strategy_group.group.nranks, self._num_of_ranks)
        _check_using_all_reduce(strategy_group.group)

    def test_data_parallel_group(self):
        dp_group = DPGroup(self._list_of_rank)
        self.assertEqual(dp_group.world_size, self._num_of_ranks)
        self.assertEqual(dp_group.group.nranks, self._num_of_ranks)
        _check_using_all_reduce(dp_group.group)

    def test_model_parallel_group(self):
        mp_group = MPGroup(self._list_of_rank)
        self.assertEqual(mp_group.world_size, self._num_of_ranks)
        self.assertEqual(mp_group.group.nranks, self._num_of_ranks)
        _check_using_all_reduce(mp_group.group)

    def test_sharding_parallel_group(self):
        sharding_group = ShardingGroup(self._list_of_rank)
        self.assertEqual(sharding_group.world_size, self._num_of_ranks)
        self.assertEqual(sharding_group.group.nranks, self._num_of_ranks)
        _check_using_all_reduce(sharding_group.group)

    def test_pipeline_parallel_group(self):
        pp_group = PPGroup(self._list_of_rank)
        send_next_group, send_prev_group, recv_next_group, recv_prev_group = pp_group.p2p_groups
        if self._global_rank == 0:
            self.assertEqual(pp_group.rank_of_next_stage, 1)
            self.assertEqual(pp_group.rank_of_prev_stage, 1)
            _check_using_send(send_next_group, self._peer_rank)
            _check_using_send(send_prev_group, self._peer_rank)
            _check_using_recv(recv_prev_group, self._peer_rank)
            _check_using_recv(recv_next_group, self._peer_rank)
        else:
            self.assertEqual(pp_group.rank_of_next_stage, 0)
            self.assertEqual(pp_group.rank_of_prev_stage, 0)
            _check_using_recv(recv_prev_group, self._peer_rank)
            _check_using_recv(recv_next_group, self._peer_rank)
            _check_using_send(send_next_group, self._peer_rank)
            _check_using_send(send_prev_group, self._peer_rank)


if __name__ == '__main__':
    unittest.main()

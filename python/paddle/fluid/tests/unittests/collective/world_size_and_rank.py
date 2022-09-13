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
import paddle
import paddle.distributed as dist


class TestWorldSizeAndRankAPI(unittest.TestCase):

    def setUp(self):
        self._num_of_ranks = 2
        self._subgroup_ranks = [0, 1]
        dist.init_parallel_env()
        self._subgroup = dist.new_group(self._subgroup_ranks)
        self._global_rank = dist.get_rank()

    def test_default_env_world_size(self):
        self.assertEqual(dist.get_world_size(), self._num_of_ranks)

    def test_given_group_world_size(self):
        world_size = 2 if self._global_rank in self._subgroup_ranks else -1
        self.assertEqual(dist.get_world_size(self._subgroup), world_size)

    def test_given_group_rank(self):
        rank = self._subgroup_ranks.index(
            self._global_rank
        ) if self._global_rank in self._subgroup_ranks else -1
        self.assertEqual(dist.get_rank(self._subgroup), rank)


if __name__ == '__main__':
    unittest.main()

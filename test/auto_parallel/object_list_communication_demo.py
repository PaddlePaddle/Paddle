# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import random

import numpy as np

import paddle
import paddle.distributed as dist


class TestObjectListCommunication:
    def init_dist_env(self):
        dist.init_parallel_env()
        paddle.seed(2025)
        np.random.seed(2025)
        random.seed(2025)

    def test_object_list_communication(self):
        """Test object list communication functionalities including parameter validation,
        group operations and normal communication process"""
        self.init_dist_env()
        curr_rank = dist.get_rank()

        # Test case 1: Parameter validation - empty list
        if curr_rank == 0:
            try:
                dist.send_object_list([], dst=1)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass
        else:
            try:
                dist.recv_object_list([], src=0)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

        # Test case 2: Group operations - rank not in group
        excluded_group = dist.new_group([2, 3])
        if curr_rank == 0:
            result = dist.send_object_list(
                ["test"], dst=1, group=excluded_group
            )
            assert result is False
        elif curr_rank == 1:
            result = dist.recv_object_list([None], src=0, group=excluded_group)
            assert result is False

        # Test case 3: Group operations - parameter conflicts
        if curr_rank == 0:
            try:
                dist.send_object_list(["test"], dst=1, group_dst=1)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass
        elif curr_rank == 1:
            try:
                dist.recv_object_list([None], src=0, group_src=0)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

        # Test case 4: Normal communication process
        if curr_rank == 0:
            data = [
                42,  # integer
                "hello",  # string
                {"key": "value"},  # dictionary
            ]
            result = dist.send_object_list(data, dst=1)
            assert result is True
        elif curr_rank == 1:
            data = [None] * 3
            result = dist.recv_object_list(data, src=0)
            assert result is True

            assert data[0] == 42
            assert data[1] == "hello"
            assert data[2] == {"key": "value"}


if __name__ == '__main__':
    TestObjectListCommunication().test_object_list_communication()

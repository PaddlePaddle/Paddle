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


import paddle
import paddle.distributed as dist


class _recv_info:
    def __init__(self, tensor, placements):
        self.obj_size = 10
        self.obj_type1 = paddle.distributed.Shard(0)
        self.obj_type2 = paddle.distributed.Replicate()
        self.obj_type3 = paddle.distributed.Partial()
        self.obj_list = [
            paddle.distributed.Shard(0),
            paddle.distributed.Replicate(),
            paddle.distributed.Partial(),
        ]
        self.dtype = paddle.int64


class TestObjectListCommunication:
    def init_dist_env(self):
        dist.init_parallel_env()
        paddle.seed(2025)

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
        send_list = ["test"]
        recv_list = [None]
        if curr_rank == 0:
            # test the dst is not in the group
            dist.send_object_list(["test"], dst=1, group=excluded_group)
        elif curr_rank == 1:
            # test the src is not in the group
            dist.recv_object_list([None], src=0, group=excluded_group)
            assert recv_list[0] is None

        excluded_group_1 = dist.new_group([0, 1])
        if curr_rank == 0:
            dist.send_object_list(send_list, dst=1, group=excluded_group_1)
        elif curr_rank == 1:
            dist.recv_object_list(recv_list, src=0, group=excluded_group_1)
            assert recv_list[0] == "test"

        # Test case 3: Group operations - parameter conflicts
        if curr_rank == 0:
            try:
                dist.send_object_list(["test"], dst=1, dst_in_group=1)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass
        elif curr_rank == 1:
            try:
                dist.recv_object_list([None], src=0, src_in_group=0)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

        # Test case 4: Group operations - using src_in_group/dst_in_group
        test_group = dist.new_group([0, 1])
        if curr_rank == 0:
            data = ["test_group_dst"]
            dist.send_object_list(data, group=test_group, dst_in_group=1)
        elif curr_rank == 1:
            data = [None]
            dist.recv_object_list(data, group=test_group, src_in_group=0)
            assert data[0] == "test_group_dst"

        # Test case 5: Normal communication process
        if curr_rank == 0:
            data = [
                42,  # integer
                "hello",  # string
                {"key": "value"},  # dictionary
            ]
            dist.send_object_list(data, dst=1)
        elif curr_rank == 1:
            data = [None] * 3
            dist.recv_object_list(data, src=0)

            assert data[0] == 42
            assert data[1] == "hello"
            assert data[2] == {"key": "value"}

        # Test case 6: Test objects with distributed attributes
        curr_rank = dist.get_rank()

        if curr_rank == 0:
            data1 = _recv_info(None, None)
            data = [data1]
            dist.send_object_list(data, dst=1)
        elif curr_rank == 1:
            data = [None]
            dist.recv_object_list(data, src=0)

            assert isinstance(data[0], _recv_info)
            assert type(data[0].obj_size) == int
            assert data[0].obj_size == 10

            assert isinstance(data[0].obj_type1, paddle.distributed.Shard)
            assert isinstance(data[0].obj_type2, paddle.distributed.Replicate)
            assert isinstance(data[0].obj_type3, paddle.distributed.Partial)

            assert data[0].dtype == paddle.int64

            assert len(data[0].obj_list) == 3
            assert isinstance(data[0].obj_list[0], paddle.distributed.Shard)
            assert isinstance(data[0].obj_list[1], paddle.distributed.Replicate)
            assert isinstance(data[0].obj_list[2], paddle.distributed.Partial)


if __name__ == '__main__':
    TestObjectListCommunication().test_object_list_communication()

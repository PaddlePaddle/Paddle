# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
from auto_parallel.hybrid_strategy.semi_auto_save_state_dict import (
    get_global_state_dict,
)

import paddle
import paddle.distributed as dist
from paddle.distributed import load_state_dict
from paddle.distributed.checkpoint.utils import (
    compute_local_shape_and_global_offset,
    get_coordinator,
)


class TestLoadStateDict:
    def __init__(self):
        self._ckpt_path = os.getenv("ckpt_path")

    def test_load_state_dict_with_one_device(self):
        global_state_dict = get_global_state_dict()
        saved_w1, saved_w2 = list(global_state_dict.values())
        w1 = paddle.zeros_like(saved_w1)
        w2 = paddle.zeros_like(saved_w2)
        state_dict = dict(zip(list(global_state_dict.keys()), [w1, w2]))
        load_state_dict(state_dict, self._ckpt_path)
        # check
        expect_w1 = saved_w1
        expect_w2 = saved_w2
        expect_state_dict = dict(
            zip(list(global_state_dict.keys()), [expect_w1, expect_w2])
        )
        for k, v in state_dict.items():
            assert k in expect_state_dict, k
            self.check_tensor_eq(v, expect_state_dict[k])

    def test_load_state_dict_with_four_devices(self):
        global_state_dict = get_global_state_dict()
        saved_w1, saved_w2 = list(global_state_dict.values())
        w1 = paddle.zeros_like(saved_w1)
        w2 = paddle.zeros_like(saved_w2)
        mesh = dist.ProcessMesh([0, 1, 2, 3])
        sharded_w1 = dist.shard_tensor(
            w1, mesh, [dist.Shard(0), dist.Replicate()]
        )
        sharded_w2 = dist.shard_tensor(
            w2, mesh, [dist.Replicate(), dist.Replicate()]
        )
        state_dict = dict(
            zip(list(global_state_dict.keys()), [sharded_w1, sharded_w2])
        )
        load_state_dict(state_dict, self._ckpt_path)
        # check
        cur_rank = paddle.distributed.get_rank()
        expect_w1 = saved_w1.split(num_or_sections=[4, 4, 4, 1], axis=0)[
            cur_rank
        ]
        expect_w2 = sharded_w2
        expect_state_dict = dict(
            zip(list(global_state_dict.keys()), [expect_w1, expect_w2])
        )
        for k, v in state_dict.items():
            assert k in expect_state_dict, k
            self.check_tensor_eq(v._local_value(), expect_state_dict[k])

    def test_load_state_dict_with_two_devices(self):
        global_state_dict = get_global_state_dict()
        saved_w1, saved_w2 = list(global_state_dict.values())
        w1 = paddle.zeros_like(saved_w1)
        w2 = paddle.zeros_like(saved_w2)
        mesh = dist.ProcessMesh([0, 1])
        sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
        sharded_w2 = dist.shard_tensor(w2, mesh, [dist.Shard(1)])
        state_dict = dict(
            zip(list(global_state_dict.keys()), [sharded_w1, sharded_w2])
        )
        load_state_dict(state_dict, self._ckpt_path)
        # check
        cur_rank = paddle.distributed.get_rank()
        expect_w1 = saved_w1.split(num_or_sections=[7, 6], axis=0)[cur_rank]
        expect_w2 = saved_w2.split(2, axis=1)[cur_rank]
        expect_state_dict = dict(
            zip(list(global_state_dict.keys()), [expect_w1, expect_w2])
        )
        for k, v in state_dict.items():
            assert k in expect_state_dict, k
            self.check_tensor_eq(v._local_value(), expect_state_dict[k])

    def test_load_state_dict_with_eight_devices(self):
        global_state_dict = get_global_state_dict()
        saved_w1, saved_w2 = list(global_state_dict.values())
        w1 = paddle.zeros_like(saved_w1)
        w2 = paddle.zeros_like(saved_w2)
        mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]])
        sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(1), dist.Shard(0)])
        sharded_w2 = dist.shard_tensor(w2, mesh, [dist.Shard(0)])
        state_dict = dict(
            zip(list(global_state_dict.keys()), [sharded_w1, sharded_w2])
        )
        load_state_dict(state_dict, self._ckpt_path)
        # check
        cur_rank = paddle.distributed.get_rank()
        local_shape, global_offset = compute_local_shape_and_global_offset(
            sharded_w1.shape,
            sharded_w1.process_mesh,
            sharded_w1.placements,
        )
        end_offset = [
            offset + length
            for offset, length in zip(global_offset, local_shape)
        ]
        expect_w1 = paddle.slice(
            saved_w1, axes=[0, 1], starts=global_offset, ends=end_offset
        )
        cur_coordinator = get_coordinator(
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), cur_rank
        )
        expect_w2 = saved_w2.split(2, axis=0)[cur_coordinator[0]]
        expect_state_dict = dict(
            zip(list(global_state_dict.keys()), [expect_w1, expect_w2])
        )
        for k, v in state_dict.items():
            assert k in expect_state_dict, k
            self.check_tensor_eq(v._local_value(), expect_state_dict[k])

    def check_tensor_eq(self, a, b, verbose=True):
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_equal(np1, np2, verbose=verbose)

    def run_test_case(self):
        device_num = int(os.getenv("device_num"))
        if device_num == 1:
            self.test_load_state_dict_with_one_device()
        elif device_num == 2:
            self.test_load_state_dict_with_two_devices()
        elif device_num == 4:
            self.test_load_state_dict_with_four_devices()
        elif device_num == 8:
            self.test_load_state_dict_with_eight_devices()
        else:
            raise ValueError("device_num should be 1, 2, 4 or 8")


if __name__ == '__main__':
    TestLoadStateDict().run_test_case()

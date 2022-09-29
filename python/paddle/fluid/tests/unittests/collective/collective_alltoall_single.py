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
import numpy as np
import paddle.distributed as dist


class TestCollectiveAllToAllSingle(unittest.TestCase):

    def setUp(self):
        assert not paddle.distributed.is_initialized(), \
            "The distributed environment has not been initialized."
        dist.init_parallel_env()
        assert paddle.distributed.is_initialized(), \
            "The distributed environment has been initialized."

        paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})

    def test_collective_alltoall_single(self):
        rank = dist.get_rank()
        size = dist.get_world_size()

        # case 1
        input = paddle.ones([size, size], dtype='int64') * rank
        output = paddle.empty([size, size], dtype='int64')
        expected_output = paddle.concat(
            [paddle.ones([1, size], dtype='int64') * i for i in range(size)])

        group = dist.new_group([0, 1])
        dist.alltoall_single(input, output, group=group)

        np.testing.assert_allclose(output.numpy(), expected_output.numpy())
        dist.destroy_process_group(group)

        # case 2
        in_split_sizes = [i + 1 for i in range(size)]
        out_split_sizes = [rank + 1 for i in range(size)]

        input = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
        output = paddle.empty([(rank + 1) * size, size], dtype='float32')
        expected_output = paddle.concat([
            paddle.ones([rank + 1, size], dtype='float32') * i
            for i in range(size)
        ])

        group = dist.new_group([0, 1])
        task = dist.alltoall_single(input,
                                    output,
                                    in_split_sizes,
                                    out_split_sizes,
                                    sync_op=False,
                                    group=group)
        task.wait()

        np.testing.assert_allclose(output.numpy(), expected_output.numpy())
        dist.destroy_process_group(group)

    def tearDown(self):
        dist.destroy_process_group()
        assert not paddle.distributed.is_initialized(), \
            "The distributed environment has been deinitialized."


if __name__ == '__main__':
    unittest.main()

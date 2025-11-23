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

import numpy as np

import paddle
import paddle.distributed as dist


class TestShutdownProcessGroupAPI:
    def __init__(self):
        dist.init_parallel_env()
        if dist.get_rank() == 0:
            self.data = paddle.to_tensor([[7, 8, 9], [10, 11, 12]])
        else:
            self.data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

    def test_shutdown_and_recreate_all(self):
        pg = paddle.distributed.new_group([0, 1])

        result_base = self.data.clone()
        dist.all_reduce(result_base, group=pg)

        paddle.distributed.shutdown_process_group()
        paddle.distributed.restart_process_group()

        result_test = self.data.clone()
        dist.all_reduce(result_test, group=pg)

        np.testing.assert_array_equal(result_base.numpy(), result_test.numpy())

    def test_shutdown_and_recreate_single(self):
        pg = paddle.distributed.new_group([0, 1])

        result_base = self.data.clone()
        dist.all_reduce(result_base, group=pg)

        paddle.distributed.shutdown_process_group(pg)
        paddle.distributed.restart_process_group(pg)

        result_test = self.data.clone()
        dist.all_reduce(result_test, group=pg)

        np.testing.assert_array_equal(result_base.numpy(), result_test.numpy())


if __name__ == "__main__":
    test_case = TestShutdownProcessGroupAPI()
    test_case.test_shutdown_and_recreate_all()
    test_case.test_shutdown_and_recreate_single()

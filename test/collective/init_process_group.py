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


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        self.config()
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]

    def config(self):
        pass

    def test_init_process_group(self):
        paddle.distributed.init_parallel_env()
        paddle.distributed.new_group()
        group = paddle.distributed.new_group([-1, -2])
        assert group.process_group is None

        group = paddle.distributed.collective.Group(-1, 2, 0, [-1, -2])
        ret = paddle.distributed.barrier(group)
        assert ret is None
        if not self._in_pir_mode:
            paddle.enable_static()
        in_tensor = paddle.empty((1, 2))
        in_tensor2 = paddle.empty((1, 2))
        paddle.distributed.broadcast(in_tensor, src=0)
        paddle.distributed.all_gather([in_tensor, in_tensor2], in_tensor)
        print("test ok\n")


if __name__ == "__main__":
    unittest.main()

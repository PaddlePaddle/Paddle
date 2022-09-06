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

import os
import paddle.distributed as dist


class InitProcessGroupTestCase():

    def __init__(self):
        self._backend = os.getenv("backend")
        self._master_addr = os.getenv("PADDLE_MASTER").split(":")[0]
        self._master_port = int(os.getenv("PADDLE_MASTER").split(":")[1])
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._is_master = (self._rank == 0)
        if self._backend not in ["nccl", "gloo"]:
            raise NotImplementedError(
                "Only support nccl and gloo as the backend for now.")

    def run_test_case(self):
        store = dist.TCPStore(self._master_addr, self._master_port,
                              self._is_master, self._world_size)
        group = dist.init_process_group(self._backend, store, self._rank,
                                        self._world_size)

        assert group.is_member() is True
        assert group.id == 0


if __name__ == "__main__":
    InitProcessGroupTestCase().run_test_case()

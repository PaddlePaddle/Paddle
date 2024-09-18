# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("../legacy_test")

from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main

import paddle
from paddle import base

paddle.enable_static()


class TestCollectiveScatterAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype="float32"):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata",
                shape=[10, 1000],
                dtype=dtype,
            )
            toutdata = paddle.tensor.fill_constant(
                shape=[5, 1000], dtype=dtype, value=1.0
            )
            tensor_list = None
            if rank == 1:
                tensor_list = paddle.split(tindata, 2, axis=0)
            paddle.distributed.scatter(toutdata, tensor_list, src=1)
            return [toutdata]


if __name__ == "__main__":
    runtime_main(TestCollectiveScatterAPI, "scatter")

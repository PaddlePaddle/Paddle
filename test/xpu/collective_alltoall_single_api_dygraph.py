# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import test_collective_api_base as test_base
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
import paddle.distributed as dist
from paddle import base


class TestCollectiveAllToAllSingleAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            if indata.dtype == "bfloat16":
                indata = convert_float_to_uint16(indata)
                tindata = paddle.to_tensor(indata)
                toutdata = paddle.empty_like(tindata)
                dist.alltoall_single(toutdata, tindata)
                return [convert_uint16_to_float(toutdata.numpy())]
            else:
                tindata = paddle.to_tensor(indata)
                toutdata = paddle.empty_like(indata)
                dist.alltoall_single(toutdata, tindata)
                return [toutdata.numpy()]


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllToAllSingleAPI, "alltoall_single")

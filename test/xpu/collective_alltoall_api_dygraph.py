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
import test_collective_api_base as test_base
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
import paddle.distributed as dist
from paddle import base


class TestCollectiveAllToAllAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            split_indata = np.split(indata, 2, axis=0)
            if indata.dtype == "bfloat16":
                indata1 = convert_float_to_uint16(split_indata[0])
                indata2 = convert_float_to_uint16(split_indata[1])

                tindata1 = paddle.to_tensor(indata1)
                tindata2 = paddle.to_tensor(indata2)
                toutdata1 = paddle.empty_like(indata1)
                toutdata2 = paddle.empty_like(indata2)

                dist.alltoall([tindata1, tindata2], [toutdata1, toutdata2])
                return [
                    convert_uint16_to_float(toutdata1.numpy()),
                    convert_uint16_to_float(toutdata2.numpy()),
                ]
            else:
                indata1 = split_indata[0]
                indata2 = split_indata[1]

                tindata1 = paddle.to_tensor(indata1)
                tindata2 = paddle.to_tensor(indata2)
                toutdata1 = paddle.empty_like(indata1)
                toutdata2 = paddle.empty_like(indata2)

                dist.alltoall([toutdata1, toutdata2], [tindata1, tindata2])
                return [toutdata1.numpy(), toutdata2.numpy()]


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllToAllAPI, "alltoall")

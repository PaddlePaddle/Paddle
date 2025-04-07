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


class TestCollectiveAllToAllSingleAPIUnequalSplitEmpty(
    test_base.TestCollectiveAPIRunnerBase
):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            if rank == 0:
                in_split_sizes = [0, 0]
                out_split_sizes = [0, 0]

                none_shape = list(indata.shape)
                none_shape[0] = 0

                indata = np.empty(none_shape, dtype=indata.dtype)
                out_shape = none_shape
            elif rank == 1:
                dim0 = indata.shape[0]
                in_split_sizes = [0, dim0]
                out_split_sizes = [0, dim0]
                out_shape = indata.shape
            else:
                raise ValueError(f"only support nranks==2, but got rank {rank}")

            if indata.dtype == "bfloat16":
                indata = convert_float_to_uint16(indata)
                tindata = paddle.to_tensor(indata)
                toutdata = paddle.empty(out_shape, dtype=tindata.dtype)
                dist.alltoall_single(
                    toutdata,
                    tindata,
                    out_split_sizes=out_split_sizes,
                    in_split_sizes=in_split_sizes,
                )
                return [convert_uint16_to_float(toutdata.numpy())]
            else:
                tindata = paddle.to_tensor(indata)
                toutdata = paddle.empty(out_shape, dtype=tindata.dtype)
                dist.alltoall_single(
                    toutdata,
                    tindata,
                    out_split_sizes=out_split_sizes,
                    in_split_sizes=in_split_sizes,
                )
                return [toutdata.numpy()]


if __name__ == "__main__":
    test_base.runtime_main(
        TestCollectiveAllToAllSingleAPIUnequalSplitEmpty,
        "alltoall_single_unequal_split_empty",
    )

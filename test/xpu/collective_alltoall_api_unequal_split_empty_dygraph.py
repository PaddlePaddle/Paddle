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


class TestCollectiveAllToAllAPIUnequalSplitEmpty(
    test_base.TestCollectiveAPIRunnerBase
):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            none_shape = list(indata.shape)
            none_shape[0] = 0

            if rank == 0:
                in_data_list = [
                    np.empty(none_shape, dtype=indata.dtype) for _ in range(2)
                ]
                out_data_list = [
                    np.empty(none_shape, dtype=indata.dtype),
                    np.empty_like(indata),
                ]
            elif rank == 1:
                in_data_list = [
                    indata,
                    np.empty(none_shape, dtype=indata.dtype),
                ]
                out_data_list = [
                    np.empty(none_shape, dtype=indata.dtype) for _ in range(2)
                ]
            else:
                raise ValueError(f"only support nranks==2, but got rank {rank}")

            # NOTE: this is a hack relying on an undocumented behavior that `to_tensor` uses uint16 to replace bfloat16
            if indata.dtype == "bfloat16":
                tindata = [
                    paddle.to_tensor(convert_float_to_uint16(data))
                    for data in in_data_list
                ]
                toutdata = [
                    paddle.to_tensor(convert_float_to_uint16(data))
                    for data in out_data_list
                ]
                dist.alltoall(toutdata, tindata)
                return [
                    convert_uint16_to_float(data.numpy()) for data in toutdata
                ]
            else:
                tindata = [paddle.to_tensor(data) for data in in_data_list]
                toutdata = [paddle.to_tensor(data) for data in out_data_list]
                dist.alltoall(toutdata, tindata)
                return [data.numpy() for data in toutdata]


if __name__ == "__main__":
    test_base.runtime_main(
        TestCollectiveAllToAllAPIUnequalSplitEmpty,
        "alltoall_unequal_split_empty",
    )

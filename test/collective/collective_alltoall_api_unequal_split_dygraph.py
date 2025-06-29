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

import legacy_test.test_collective_api_base as test_base

import paddle
import paddle.distributed as dist
from paddle import base


class TestCollectiveAllToAllAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            dim0 = indata.shape[0]
            dim1 = indata.shape[1]
            half_dim0 = dim0 // 2
            half_dim1 = dim1 // 2

            if rank == 0:
                in_data_list = [
                    indata[: half_dim0 - 1, : half_dim1 - 1],
                    indata[: half_dim0 - 1, half_dim1 - 1 :],
                ]
                out_data_shape_list = [
                    (half_dim0 - 1, half_dim1 - 1),
                    (half_dim0 + 1, half_dim1 - 2),
                ]
            elif rank == 1:
                in_data_list = [
                    indata[half_dim0 - 1 :, : half_dim1 - 2],
                    indata[half_dim0 - 1 :, half_dim1 - 2 :],
                ]
                out_data_shape_list = [
                    (half_dim0 - 1, half_dim1 + 1),
                    (half_dim0 + 1, half_dim1 + 2),
                ]
            else:
                raise ValueError(f"only support nranks==2, but got rank {rank}")

            # NOTE: this is a hack relying on an undocumented behavior that `to_tensor` uses uint16 to replace bfloat16
            if indata.dtype == "bfloat16":
                tindata = [
                    paddle.to_tensor(data, "float32").cast("uint16")
                    for data in in_data_list
                ]
                toutdata = [
                    paddle.empty(shape, dtype="uint16")
                    for shape in out_data_shape_list
                ]
                dist.alltoall(toutdata, tindata)
                return [data.cast("float32").numpy() for data in toutdata]
            else:
                tindata = [paddle.to_tensor(data) for data in in_data_list]
                toutdata = [
                    paddle.empty(shape, dtype=tindata[0].dtype)
                    for shape in out_data_shape_list
                ]
                dist.alltoall(toutdata, tindata)
                return [data.numpy() for data in toutdata]


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllToAllAPI, "alltoall")

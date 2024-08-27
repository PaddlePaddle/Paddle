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

import numpy as np
from legacy_test.test_collective_api_base import (
    TestCollectiveAPIRunnerBase,
    runtime_main,
)

import paddle
from paddle import base
from paddle.distributed import fleet

paddle.enable_static()


class TestParallelEmbeddingAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype="float32"):
        with base.program_guard(main_prog, startup_program):
            fleet.init(is_collective=True)
            np.random.seed(2020)
            # (num_embeddings, embedding_dim) = (12, 8)
            size = (12, 8)
            np_array = np.random.rand(size[0], size[1])
            paddle.seed(2020)
            data_in = paddle.randint(0, size[0], shape=(10, 4))

            data = paddle.static.data(
                name='tindata', shape=[10, 1000], dtype=dtype
            )
            per_part_size = size[0] // 2
            if rank == 0:
                param_attr = paddle.base.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(
                        np_array[0:per_part_size, :]
                    ),
                )
            else:
                param_attr = paddle.base.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(
                        np_array[per_part_size : size[0], :]
                    ),
                )

            emb_out = paddle.distributed.split(
                data_in,
                size,
                operation="embedding",
                num_partitions=2,
                weight_attr=param_attr,
            )

            return [data_in, emb_out]


if __name__ == "__main__":
    runtime_main(TestParallelEmbeddingAPI, "parallel_embedding")

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

import legacy_test.test_collective_api_base as test_base

import paddle
from paddle import base


class TestCollectiveAllgatherObjectAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with base.program_guard(main_prog, startup_program):
            # Run the collective twice, once per supported initialization
            # style, and assert the results match. This locks in the
            # alignment with torch.distributed.all_gather_object: both an
            # empty list (Paddle legacy) and a pre-allocated [None]*world_size
            # (PyTorch) must produce identical output. Doubles the comm cost
            # for this test but stays well inside the 120s timeout.
            paddle_style = []
            paddle.distributed.all_gather_object(paddle_style, indata)

            world_size = paddle.distributed.get_world_size()
            torch_style = [None for _ in range(world_size)]
            paddle.distributed.all_gather_object(torch_style, indata)

            assert paddle_style == torch_style, (
                f"all_gather_object initialization styles disagree: "
                f"empty-list {paddle_style!r} vs pre-allocated {torch_style!r}"
            )
            return torch_style


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllgatherObjectAPI, "allgather_object")

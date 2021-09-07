# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid
import unittest
import paddle.fluid.layers as layers
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main


class TestCollectiveGlobalGatherAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with fluid.program_guard(main_prog, startup_program):
            seed = os.getpid()
            np.random.seed(seed)
            in_feat = 2
            n_expert = 2
            world_size = 2
            tot_expert = n_expert * world_size
            local_expert_count = np.random.randint(
                1, 4, size=tot_expert).astype("int")
            local_expert_count = paddle.to_tensor(local_expert_count)
            global_expert_count = []
            paddle.distributed.alltoall(
                paddle.split(
                    local_expert_count, 2, axis=0),
                global_expert_count)
            global_expert_count = paddle.concat(global_expert_count, axis=0)
            fwd_expert_count = sum(global_expert_count)
            np.random.seed(seed)
            local_input_buf = np.random.rand(fwd_expert_count,
                                             in_feat).astype("float32")
            local_input_buf = paddle.to_tensor(local_input_buf)
            local_input_buf.stop_gradient = False
            output = paddle.distributed.utils.global_gather(
                local_input_buf, local_expert_count, global_expert_count)
            output.stop_gradient = False
            c = output * output
            c.stop_gradient = False
            c.backward()
            return [output.numpy(), local_input_buf.grad.numpy()]


if __name__ == "__main__":
    runtime_main(TestCollectiveGlobalGatherAPI, "global_gather")

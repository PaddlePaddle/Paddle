# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
)


class TestFusedCommBufferGradChecker(unittest.TestCase):
    def test_fused_comm_buffer_grad_checker(self):
        linear = paddle.nn.Linear(10, 10)
        w = linear.weight
        b = linear.bias
        w.main_grad = None
        b.main_grad = None
        buffer = FusedCommBuffer(
            id=0,
            params=[w, b],
            comm_group=None,
            acc_steps=10,
            act=HOOK_ACTION.ALL_REDUCE,
        )
        assert buffer.use_main_grad
        buffer.add_grad(w)
        buffer.add_grad(b)
        w.main_grad = paddle.to_tensor([1], stop_gradient=True, dtype="float32")
        try:
            buffer.add_grad(w)
            raise AssertionError(
                "Above add_grad should raise value error, this assertion should be unreachable."
            )
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main()

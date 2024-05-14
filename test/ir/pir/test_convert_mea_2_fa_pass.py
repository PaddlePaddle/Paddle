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

import unittest

import numpy as np
from fused_pass.pass_test import PassTest

import paddle
from paddle.base import core
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)

paddle.enable_static()


class TestConvertMEA2FA(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                q = paddle.static.data(
                    name='q', shape=[2, 8, 32, 128], dtype="float16"
                )
                k = paddle.static.data(
                    name='k', shape=[2, 8, 32, 128], dtype="float16"
                )
                v = paddle.static.data(
                    name='v', shape=[2, 8, 32, 128], dtype="float16"
                )

                out, _ = memory_efficient_attention(q, k, v, training=False)
                self.pass_attr_list = [{'convert_MEA_to_FA': {}}]
                self.feeds = {
                    "q": np.random.random((2, 8, 32, 128)).astype("float16"),
                    "k": np.random.random((2, 8, 32, 128)).astype("float16"),
                    "v": np.random.random((2, 8, 32, 128)).astype("float16"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.flash_attn": 1}
                yield [main_prog, start_prog], False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()

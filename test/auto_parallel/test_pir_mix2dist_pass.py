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
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

paddle.enable_static()

BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_SIZE = 8
MP_SIZE = 2


class TestBuildFakeProgram(unittest.TestCase):
    def test_build_api(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                w0 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[HIDDEN_SIZE, HIDDEN_SIZE],
                    name="w0",
                    initializer=paddle.nn.initializer.Uniform(),
                )

        apply_mix2dist_pass(main_program)


if __name__ == "__main__":
    unittest.main()

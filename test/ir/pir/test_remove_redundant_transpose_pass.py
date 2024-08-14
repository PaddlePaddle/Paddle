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

import os
import unittest

import numpy as np
from fused_pass.pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()


class TestRemoveRedundantTransposePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for perm1_shape in [[1, 2, 0]]:
            for perm2_shape in [[0, 2, 1]]:
                with paddle.pir_utils.IrGuard():
                    main_prog = paddle.static.Program()
                    start_prog = paddle.static.Program()
                    with paddle.pir.core.program_guard(main_prog, start_prog):
                        x = paddle.static.data(
                            name='x', shape=[2, 3, 4], dtype="float32"
                        )
                        out = paddle.transpose(
                            paddle.transpose(x, perm1_shape), perm2_shape
                        )
                        out = paddle.assign(out)
                        self.pass_attr_list = [
                            {'remove_redundant_transpose_pass': {}}
                        ]
                        self.feeds = {
                            "x": np.random.random((2, 3, 4)).astype("float32")
                        }
                        self.fetch_list = [out]
                        self.valid_op_map = {"pd_op.transpose": 1}
                        yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct()

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()

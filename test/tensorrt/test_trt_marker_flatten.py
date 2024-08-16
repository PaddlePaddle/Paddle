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
from pass_test import PassTest

import paddle
from paddle.base import core


class TestFlattenTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            for x_shape in [[2, 1, 1, 19]]:
                with paddle.pir.core.program_guard(main_prog, start_prog):
                    x = paddle.static.data(
                        name='x', shape=x_shape, dtype='float32'
                    )
                    flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
                    flatten_out = flatten(
                        paddle.transpose(x, perm=[0, 3, 1, 2])
                    )
                    out = paddle.assign(flatten_out)
                    self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                    self.feeds = {
                        "x": np.random.random(x_shape).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.fusion_transpose_flatten_concat": 0,
                    }
                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.trt_expected_ops = {"pd_op.flatten"}

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()

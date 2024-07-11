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
from paddle.pir.core import create_parameter


class TestTRTPattern(PassTest):
    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        for bias_shape in [[1, 32, 1, 1], [32, 1, 1], [32]]:
            with paddle.pir_utils.IrGuard():
                main_prog = paddle.static.Program()
                start_prog = paddle.static.Program()
                with paddle.pir.core.program_guard(main_prog, start_prog):
                    x = paddle.static.data(
                        name='x', shape=[3, 1, 28, 28], dtype='float32'
                    )
                    conv2d = paddle.nn.Conv2D(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        data_format='NCHW',
                        bias_attr=False,
                    )

                    y = create_parameter(
                        name="y",
                        shape=bias_shape,
                        dtype='float32',
                        initializer=paddle.nn.initializer.Assign(
                            np.random.random(bias_shape).astype("float32")
                        ),
                    )
                    act_op = paddle.nn.ReLU()
                    act_out = act_op(paddle.add(conv2d(x), y))
                    pool2d = paddle.nn.MaxPool2D(
                        kernel_size=2, stride=2, padding=0
                    )
                    padding_out = pool2d(act_out)
                    softmax = paddle.nn.Softmax()
                    softmax_out = softmax(padding_out)
                    reshaped_out = paddle.reshape(
                        softmax_out, [softmax_out.shape[0], -1]
                    )
                    out = paddle.assign(reshaped_out)
                    self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                    self.feeds = {
                        "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.fused_conv2d_add_act": 0,
                    }
                    return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()

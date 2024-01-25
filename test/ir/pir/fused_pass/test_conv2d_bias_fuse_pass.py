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

import numpy as np
from pass_test import PassTest

import paddle

paddle.enable_static()


class TestConv2dAddFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[8, 6, 7, 8], dtype='float32'
                )
                y = paddle.static.data(name="y", shape=[5], dtype="float32")
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=6,
                    out_channels=2,
                    kernel_size=[1, 1],
                    groups=2,
                    stride=[3, 3],
                    padding=[4, 1, 3, 4],
                    dilation=[2, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = paddle.add(conv2d(x), y)
                out = paddle.assign(out)
                self.pass_list = ['conv2d_bias_fuse_pass']
                self.feeds = {
                    "x": np.random.random((8, 6, 7, 8)).astype("float32"),
                    "y": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.conv2d": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        # todo(bukejiyu): This pass will support accuracy verification in the future
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()

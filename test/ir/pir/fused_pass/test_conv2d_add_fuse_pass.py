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
from paddle.base import core
from paddle.pir.core import create_parameter


class TestConv2dAddFusePass(PassTest):
    r"""
    x_var   filter(w)
      \       /
         conv2d  bias(w)
           |    /
            add
             |
            out_var
    """

    def is_program_valid(self, program=None):
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
                    bias = create_parameter(
                        name="bias",
                        shape=bias_shape,
                        dtype='float32',
                        initializer=paddle.nn.initializer.Assign(
                            np.random.random(bias_shape).astype("float32")
                        ),
                    )
                    conv2d = paddle.nn.Conv2D(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        padding="SAME",
                        data_format='NCHW',
                        bias_attr=False,
                    )
                    out = paddle.add(conv2d(x), bias)
                    out = paddle.assign(out)
                    self.pass_attr_list = [{'conv2d_add_fuse_pass': {}}]
                    self.feeds = {
                        "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.fused_conv2d_add_act": 1,
                        "pd_op.conv2d": 0,
                        "pd_op.add": 0,
                    }
                    return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dAddFusePass_cutlass(PassTest):
    r"""
      x_var   f_var(w)
    \       /
       conv2d
         |
      conv2d_var    y_var(w)
          \          /
         elementwise_add
              |
            add_var
    """

    def is_program_valid(self, program):
        return True

    def sample_program(self):
        for dtype in ["float16"]:
            for w_shape in [[32, 8, 3, 3]]:
                for bias_shape in [[1, 1, 1, 32], [1, 1, 32], [32]]:
                    rand_value = (
                        0.001 * paddle.rand(shape=w_shape, dtype=dtype).numpy()
                    )
                    with paddle.pir_utils.IrGuard():
                        main_prog = paddle.static.Program()
                        start_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=[3, 28, 28, 8], dtype='float16'
                            )
                            w = create_parameter(
                                shape=w_shape,
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            conv2d_out = paddle.nn.functional.conv2d(
                                x=x,
                                weight=w,
                                bias=None,
                                padding=1,
                                data_format="NHWC",
                            )

                            y = create_parameter(
                                name="y",
                                shape=bias_shape,
                                dtype='float16',
                                initializer=paddle.nn.initializer.Assign(
                                    np.random.random(bias_shape).astype(
                                        "float16"
                                    )
                                ),
                            )
                            out = paddle.add(conv2d_out, y)
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {'conv2d_add_fuse_pass': {"use_cutlass": True}}
                            ]
                            self.feeds = {
                                "x": np.random.random((3, 28, 28, 8)).astype(
                                    "float16"
                                ),
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.add": 0,
                                "pd_op.conv2d": 0,
                                "pd_op.fused_conv2d_add_act": 1,
                            }
                            yield [main_prog, start_prog], False

    def setUp(self):
        self.use_cutlass = False
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.skip_accuracy_verification = False

    def test_check_output(self):
        """
        conv2d_add_fuse_pass's unittest have been tested locally, this pass
        relies on users manually installing libCutlassConv2d.so,
        this test code has been temporarily shut down(i.e. self.use_cutlass = False).
        You can easily run this unittest by manually installing libCutlassConv2d.so and set self.use_cutlass True.
        (See details: paddle/phi/kernels/fusion/cutlass/conv2d/README.md)
        """
        if self.use_cutlass:
            self.check_pass_correct(1e-3, 1e-3)


if __name__ == "__main__":
    unittest.main()

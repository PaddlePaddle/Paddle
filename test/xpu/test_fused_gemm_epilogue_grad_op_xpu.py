# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core


def get_outputs(DOut, X, Y):
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)

    return DX, DY, DBias


class XPUTestFuseGemmGradOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fused_gemm_epilogue_grad'
        self.use_dynamic_create_class = False

    class TestFuseGemmEpilogueGradOp(XPUOpTest):
        def setUp(self):
            paddle.enable_static()
            self.op_type = "fused_gemm_epilogue_grad"
            self.__class__.no_need_check_grad = True

            self.dtype = self.in_type
            self.init_shape()
            self.init_data()
            self.init_output()

        def init_shape(self):
            self.x_shape = (8, 4)
            self.y_shape = (4, 128)
            self.out_shape = (8, 128)

        def init_data(self):
            self.dout = np.random.uniform(-0.5, 0.5, self.out_shape)
            self.x = np.random.uniform(-0.5, 0.5, self.x_shape)
            self.y = np.random.uniform(-0.5, 0.5, self.y_shape)

            if self.dtype != np.uint16:
                self.dout = self.dout.astype(self.dtype)
                self.x = self.x.astype(self.dtype)
                self.y = self.y.astype(self.dtype)

            # numpy outputs
            self.dx, self.dy, self.dbias = get_outputs(
                self.dout, self.x, self.y
            )

            # special for bfloat16
            if self.dtype == np.uint16:
                self.dout = convert_float_to_uint16(self.dout)
                self.x = convert_float_to_uint16(self.x)
                self.y = convert_float_to_uint16(self.y)
                self.dx = convert_float_to_uint16(self.dx)
                self.dy = convert_float_to_uint16(self.dy)
                self.dbias = convert_float_to_uint16(self.dbias)

            # inputs
            self.inputs = {
                'DOut': self.dout,
                'X': self.x,
                'Y': self.y,
            }

            # attrs
            self.attrs = {"activation_grad": 'none'}

        def init_output(self):
            self.outputs = {'DX': self.dx, 'DY': self.dy, 'DBias': self.dbias}

        def test_check_output(self):
            self.atol = 5e-4
            if self.dtype == np.float16:
                self.atol = 1e-3
            self.check_output_with_place(core.XPUPlace(0), atol=self.atol)

    class TestFuseGemmEpilogueGradOp2(TestFuseGemmEpilogueGradOp):
        def init_output(self):
            self.outputs = {'DY': self.dy, 'DBias': self.dbias}

    class TestFuseGemmEpilogueGradOp3(XPUOpTest):
        def init_output(self):
            self.outputs = {
                'DY': self.dy,
            }

    class TestFuseGemmEpilogueGradOp4(XPUOpTest):
        def init_output(self):
            self.outputs = {
                'DX': self.dx,
                'DY': self.dy,
            }


support_types = get_xpu_op_support_types('fused_gemm_epilogue_grad')
for stype in support_types:
    create_test_class(globals(), XPUTestFuseGemmGradOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()

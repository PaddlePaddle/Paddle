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

import sys

sys.path.append("..")

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


def get_outputs(DOut, X, Y):
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)

    return DX, DY, DBias


class XPUTestFuseGemmGradOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'fused_gemm_epilogue_grad'
        self.use_dynamic_create_class = False

    class TestFuseGemmEpilogueGradOpDXYBias1(XPUOpTest):

        def setUp(self):
            paddle.enable_static()
            self.op_type = "fused_gemm_epilogue_grad"
            self.__class__.no_need_check_grad = True

            self.dtype = self.in_type
            self.init_data()

        def init_data(self):
            self.inputs = {
                'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
                'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
                'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
            }

            self.attrs = {"activation": 'none'}

            DX, DY, DBias = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                        self.inputs['Y'])
            self.outputs = {'DX': DX, 'DY': DY, 'DBias': DBias}

        def test_check_output(self):
            self.atol = 1e-4
            if self.dtype == np.float16:
                self.atol = 1e-3
            self.check_output_with_place(core.XPUPlace(0), atol=self.atol)

    class TestFuseGemmEpilogueGradOpDXYBias2(XPUOpTest):

        def init_data(self):
            self.inputs = {
                'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
                'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
                'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
            }

            self.attrs = {"activation": 'none'}

            _, DY, DBias = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                       self.inputs['Y'])
            self.outputs = {'DY': DY, 'DBias': DBias}

    class TestFuseGemmEpilogueGradOpDXYBias3(XPUOpTest):

        def init_data(self):
            self.inputs = {
                'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
                'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
                'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
            }

            self.attrs = {"activation": 'none'}

            _, DY, _ = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                   self.inputs['Y'])
            self.outputs = {'DY': DY}

    class TestFuseGemmEpilogueGradOpDXYBias4(XPUOpTest):

        def init_data(self):
            self.inputs = {
                'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
                'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
                'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
            }

            self.attrs = {"activation": 'none'}

            DX, DY, _ = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                    self.inputs['Y'])
            self.outputs = {'DX': DX, 'DY': DY}


support_types = get_xpu_op_support_types('fused_gemm_epilogue_grad')
for stype in support_types:
    create_test_class(globals(), XPUTestFuseGemmGradOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()

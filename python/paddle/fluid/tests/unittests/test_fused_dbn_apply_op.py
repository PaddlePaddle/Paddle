#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid.core as core


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
    )


skip_msg = "only support with cuda and Ampere or later devices"


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDbnApplyOp(OpTest):
    def setUp(self):
        self.op_type = "fused_dbn_apply"
        self.dtype = np.float16
        self.math_type = np.float32
        self.outputs = None

        self.init_test_case()
        self.init_attr()

        c_dim = self.input_size[-1]
        dY_input = np.random.random(self.input_size).astype(self.dtype)
        X_input = np.random.random(self.input_size).astype(self.dtype)
        A_input = np.random.random(c_dim).astype(self.math_type)
        B_input = np.random.random(c_dim).astype(self.math_type)
        C_input = np.random.random(c_dim).astype(self.math_type)

        X_dual_input = np.random.random(self.input_size).astype(self.dtype)
        A_dual_input = np.random.random(c_dim).astype(self.math_type)
        B_dual_input = np.random.random(c_dim).astype(self.math_type)
        C_dual_input = np.random.random(c_dim).astype(self.math_type)

        dX_output = (
            dY_input.astype(self.math_type) * A_input.reshape((1, 1, 1, c_dim))
            + X_input.astype(self.math_type) * B_input.reshape((1, 1, 1, c_dim))
            + C_input.reshape((1, 1, 1, c_dim))
        )
        dX_output = dX_output.astype(self.dtype)

        dX_dual_output = (
            dY_input.astype(self.math_type)
            * A_dual_input.reshape((1, 1, 1, c_dim))
            + X_dual_input.astype(self.math_type)
            * B_dual_input.reshape((1, 1, 1, c_dim))
            + C_dual_input.reshape((1, 1, 1, c_dim))
        )
        dX_dual_output = dX_dual_output.astype(self.dtype)

        self.attrs = {'fuse_dual': self.fuse_dual}

        if self.fuse_dual:
            self.inputs = {
                'dY': OpTest.np_dtype_to_fluid_dtype(dY_input),
                'X': OpTest.np_dtype_to_fluid_dtype(X_input),
                'A': OpTest.np_dtype_to_fluid_dtype(A_input),
                'B': OpTest.np_dtype_to_fluid_dtype(B_input),
                'C': OpTest.np_dtype_to_fluid_dtype(C_input),
                'X_dual': OpTest.np_dtype_to_fluid_dtype(X_dual_input),
                'A_dual': OpTest.np_dtype_to_fluid_dtype(A_dual_input),
                'B_dual': OpTest.np_dtype_to_fluid_dtype(B_dual_input),
                'C_dual': OpTest.np_dtype_to_fluid_dtype(C_dual_input),
            }
        else:
            self.inputs = {
                'dY': OpTest.np_dtype_to_fluid_dtype(dY_input),
                'X': OpTest.np_dtype_to_fluid_dtype(X_input),
                'A': OpTest.np_dtype_to_fluid_dtype(A_input),
                'B': OpTest.np_dtype_to_fluid_dtype(B_input),
                'C': OpTest.np_dtype_to_fluid_dtype(C_input),
            }

        if self.fuse_dual:
            self.outputs = {'dX': dX_output, 'dX_dual': dX_dual_output}

        else:
            self.outputs = {'dX': dX_output}

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=2e-2)

    def init_test_case(self):
        self.input_size = [8, 32, 32, 16]  # NHWC

    def init_attr(self):
        self.fuse_dual = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDbnApplyOpDual(TestFusedDbnApplyOp):
    def init_attr(self):
        self.fuse_dual = True


if __name__ == '__main__':
    unittest.main()

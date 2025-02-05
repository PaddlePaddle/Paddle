#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle.base import core


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
    )


skip_msg = "only support with cuda and Ampere or later devices"


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasAddReluOp(OpTest):
    def setUp(self):
        self.op_type = "fused_scale_bias_add_relu"
        self.dtype = np.float16
        self.outputs = None

        self.init_test_case()
        self.init_attr()

        self.attrs = {
            'fuse_dual': self.fuse_dual,
            'exhaustive_search': self.exhaustive_search,
        }

        c_dim = self.input_size[-1]
        x1_input = np.random.random(self.input_size).astype(self.dtype) - 0.5
        x2_input = np.random.random(self.input_size).astype(self.dtype) - 0.5
        scale1_input = np.random.random(c_dim).astype(self.dtype) - 0.5
        scale2_input = np.random.random(c_dim).astype(self.dtype) - 0.5
        bias1_input = np.random.random(c_dim).astype(self.dtype) - 0.5
        bias2_input = np.random.random(c_dim).astype(self.dtype) - 0.5

        # calculate reference output
        reshaped_scale1_input = scale1_input.reshape(1, 1, 1, c_dim)
        reshaped_scale2_input = scale2_input.reshape(1, 1, 1, c_dim)
        reshaped_bias1_input = bias1_input.reshape(1, 1, 1, c_dim)
        reshaped_bias2_input = bias2_input.reshape(1, 1, 1, c_dim)

        after_bias1 = x1_input * reshaped_scale1_input + reshaped_bias1_input
        after_bias2 = x2_input * reshaped_scale2_input + reshaped_bias2_input

        if self.fuse_dual:
            after_add = after_bias1 + after_bias2
        else:
            after_add = after_bias1 + x2_input

        y_output = np.maximum(after_add, 0).astype(self.dtype)

        if self.fuse_dual:
            self.inputs = {
                'x1': x1_input,
                'scale1': scale1_input,
                'bias1': bias1_input,
                'x2': x2_input,
                'scale2': scale2_input,
                'bias2': bias2_input,
            }
        else:
            self.inputs = {
                'x1': x1_input,
                'scale1': scale1_input,
                'bias1': bias1_input,
                'x2': x2_input,
            }

        self.outputs = {
            'out': y_output,
        }

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_dygraph=False, atol=2e-2)

    def init_test_case(self):
        self.input_size = [2, 8, 8, 16]

    def init_attr(self):
        self.fuse_dual = False
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasAddReluOpDual(TestFusedScaleBiasAddReluOp):
    def init_attr(self):
        self.fuse_dual = True
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasAddReluOpExhaustive(TestFusedScaleBiasAddReluOp):
    def init_attr(self):
        self.fuse_dual = False
        self.exhaustive_search = True


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()

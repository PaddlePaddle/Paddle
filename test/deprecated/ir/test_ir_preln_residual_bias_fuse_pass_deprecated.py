#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

sys.path.append("../../ir")
from pass_test import PassTest

import paddle


class PrelnResidualBiasFusePassTest(PassTest):
    def setUp(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            self.main_program, self.startup_program
        ):
            x = paddle.static.data(name="x", shape=[128, 768], dtype="float32")
            bias = paddle.static.create_parameter(shape=[768], dtype='float32')
            y = paddle.static.data(name="y", shape=[128, 768], dtype="float32")
            x = x + bias
            elementwise_out = x + y
            out = paddle.static.nn.layer_norm(input=elementwise_out)

        self.fetch_list = [out, elementwise_out]
        self.pass_names = "preln_residual_bias_fuse_pass"
        self.fused_op_type = "fused_bias_dropout_residual_layer_norm"
        self.num_fused_ops = 1
        # self.graph_attrs = {
        #     "embedding_eltwise_layernorm_fuse_pass_flag": True,
        #     "multihead_matmul_fuse_pass_flag": True
        # }

    def test_check_program(self):
        use_gpu_set = [False]
        if paddle.device.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            opt_program = self._apply_ir_passes()
            self.check_program(opt_program)


class PrelnResidualBiasFusePassNoBiasTest(PassTest):
    def setUp(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            self.main_program, self.startup_program
        ):
            x = paddle.static.data(name="x", shape=[128, 768], dtype="float32")
            y = paddle.static.data(name="y", shape=[128, 768], dtype="float32")
            elementwise_out = x + y
            out = paddle.static.nn.layer_norm(input=elementwise_out)

        self.fetch_list = [out, elementwise_out]
        self.pass_names = "preln_residual_bias_fuse_pass"
        self.fused_op_type = "fused_bias_dropout_residual_layer_norm"
        self.num_fused_ops = 1

    def test_check_program(self):
        use_gpu_set = [False]
        if paddle.device.is_compiled_with_cuda():
            use_gpu_set.append(True)
        for use_gpu in use_gpu_set:
            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
            opt_program = self._apply_ir_passes()
            self.check_program(opt_program)


if __name__ == "__main__":
    unittest.main()

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

import paddle

paddle.pir.register_paddle_dialect()

paddle.enable_static()

ffn_patern_program = '''
{
    (%0) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<4096x8192xbf16>
    (%1) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<8192x14336xbf16>
    (%2) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<8192x14336xbf16>
    (%3) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<4096x14336xbf16>
    (%4) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<4096x14336xbf16>
    (%5) = "pd_op.matmul" (%0, %1) {kernel_name:"matmul",op_name:"pd_op.matmul",persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4096x8192xbf16>, builtin.tensor<8192x14336xbf16>) -> builtin.tensor<4096x14336xbf16>
    (%6) = "pd_op.matmul" (%0, %2) {kernel_name:"matmul",op_name:"pd_op.matmul",persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4096x8192xbf16>, builtin.tensor<8192x14336xbf16>) -> builtin.tensor<4096x14336xbf16>
    (%7) = "pd_op.swiglu" (%5, %6) {kernel_name:"swiglu",op_name:"pd_op.swiglu",persistable:[false],stop_gradient:[false]} : (builtin.tensor<4096x14336xbf16>, builtin.tensor<4096x14336xbf16>) -> builtin.tensor<4096x14336xbf16>
    (%8) = "pd_op.matmul_grad" (%0, %1, %3) {kernel_name:"matmul",op_name:"pd_op.matmul",persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4096x8192xbf16>, builtin.tensor<8192x14336xbf16>, builtin.tensor<4096x14336xbf16>) -> builtin.tensor<4096x14336xbf16>
    (%9) = "pd_op.matmul_grad" (%0, %2, %4) {kernel_name:"matmul",op_name:"pd_op.matmul",persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4096x8192xbf16>, builtin.tensor<8192x14336xbf16>, builtin.tensor<4096x14336xbf16>) -> builtin.tensor<4096x14336xbf16>
}
'''


class TestTensorFusionPass(unittest.TestCase):
    def test_tensor_fusion_pass(self):

        pir_program = paddle.pir.parse_program(ffn_patern_program)
        print("before pass")
        print(pir_program)

        pm = paddle.pir.PassManager()
        pm.add_pass('tensor_fusion_pass', {})
        pm.run(pir_program)

        print("after pass")
        print(pir_program)


if __name__ == "__main__":
    unittest.main()

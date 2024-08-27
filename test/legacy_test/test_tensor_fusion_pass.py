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

patern_program1 = '''
{
    (%0) = "builtin.parameter" () {parameter_name:"embedding_0.w_0.dist"} : () -> builtin.tensor<1x4096x4096xf32>
    (%1) = "builtin.parameter" () {parameter_name:"embedding_1.w_0.dist"} : () -> builtin.tensor<1x4096x4096xf32>
    (%2) = "pd_op.cast" (%0) {dtype:(pd_op.DataType)bfloat16} : (builtin.tensor<1x4096x4096xf32>) -> builtin.tensor<1x4096x4096xbf16>
    (%3) = "pd_op.cast" (%1) {dtype:(pd_op.DataType)bfloat16} : (builtin.tensor<1x4096x4096xf32>) -> builtin.tensor<1x4096x4096xbf16>
    (%4) = "pd_op.swiglu" (%2, %3) {} : (builtin.tensor<1x4096x4096xbf16>, builtin.tensor<1x4096x4096xbf16>) -> builtin.tensor<1x4096x4096xbf16>
}
'''


class TestTensorFusionPass(unittest.TestCase):
    def test_tensor_fusion_pass(self):
        pir_program = paddle.pir.parse_program(patern_program1)
        print("before pass")
        print(pir_program)

        pm = paddle.pir.PassManager()
        pm.add_pass('tensor_fusion_pass', {})
        pm.run(pir_program)

        print("after pass")
        print(pir_program)


if __name__ == "__main__":
    unittest.main()

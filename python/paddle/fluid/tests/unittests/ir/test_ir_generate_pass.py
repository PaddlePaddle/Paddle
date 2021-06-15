#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.ir_pass as ir_pass

paddle.enable_static()


@ir_pass.RegisterPass
def test_generate_pass():
    # pattern
    X, Y, mul_out = ir_pass.CreateVars("X", "Y", "out")
    mul = ir_pass.Op("mul")([X, Y], [mul_out])
    bias, ewadd_out = ir_pass.CreateVars("bias", "out")
    ewadd = ir_pass.Op("elementwise_add")([mul_out("X"), bias], [ewadd_out])
    # algebra
    fc = ir_pass.Op("fc")([X("Input"), Y("W"), bias("Bias")],
                          [ewadd_out("out")])
    return ir_pass.CreatePassPair([mul, ewadd], [fc])


class FCFusePassTest(unittest.TestCase):
    def test_check_pass(self):
        ir_pass.UsePass("test_generate_pass")
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            data = fluid.data(name="data", shape=[32, 32], dtype="float32")
            relu_out = fluid.layers.relu(x=data)
            weight = fluid.data(name="weight", shape=[32, 32])
            mul_out = fluid.layers.mul(relu_out,
                                       weight,
                                       x_num_col_dims=1,
                                       y_num_col_dims=1)
            bias = fluid.data(name="bias", shape=[32, 32])
            ewadd_out = fluid.layers.elementwise_add(mul_out, bias)

        graph = core.Graph(main_program.desc)
        before_nodes_num = len(graph.nodes())
        test_pass = core.get_pass("test_generate_pass")
        test_pass.apply(graph)
        after_nodes_num = len(graph.nodes())
        self.assertEqual(before_nodes_num, after_nodes_num + 6)

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import six
import paddle
from paddle import fluid

paddle.enable_static()


class GraphToProgramPass(unittest.TestCase):
    def setUp(self):
        self.origin_program = build_program()
        ir_graph = program_to_IRGraph(self.origin_program)
        self.converted_program = IRGraph_to_program(ir_graph)

    def test_check_parameter(self):
        origin_parameter = sorted(
            self.origin_program.all_parameters(), key=lambda p: p.name)
        converted_parameter = sorted(
            self.converted_program.all_parameters(), key=lambda p: p.name)

        self.assertEqual(len(origin_parameter), len(converted_parameter))

        for i in range(len(origin_parameter)):
            o_para = origin_parameter[i]
            c_para = converted_parameter[i]
            self.assertEqual(o_para.name, c_para.name)
            self.assertEqual(o_para.is_parameter, c_para.is_parameter)

    def test_check_stop_gradient(self):
        origin_vars = []
        for var in self.origin_program.list_vars():
            origin_vars.append(var)
        origin_vars = sorted(origin_vars, key=lambda v: v.name)

        converted_vars = []
        for var in self.converted_program.list_vars():
            converted_vars.append(var)
        converted_vars = sorted(converted_vars, key=lambda v: v.name)

        self.assertEqual(len(origin_vars), len(converted_vars))

        for i in range(len(origin_vars)):
            o_var = origin_vars[i]
            c_var = converted_vars[i]
            self.assertEqual(o_var.name, c_var.name)
            self.assertEqual(o_var.stop_gradient, c_var.stop_gradient)


def build_program():
    program = fluid.default_main_program()
    with fluid.program_guard(program):
        data = fluid.data(name='x', shape=[None, 13], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
    return program


def program_to_IRGraph(program):
    graph = fluid.core.Graph(program.desc)
    ir_graph = fluid.framework.IrGraph(graph, for_test=False)
    return ir_graph


def IRGraph_to_program(ir_graph):
    return ir_graph.to_program()


if __name__ == "__main__":
    unittest.main()

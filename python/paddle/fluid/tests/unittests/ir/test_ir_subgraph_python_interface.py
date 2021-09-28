# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import six

from paddle.fluid.framework import IrGraph
from paddle.fluid.framework import IrNode
from paddle.fluid import core
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard, default_startup_program
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass

paddle.enable_static()


class TestQuantizationSubGraph(unittest.TestCase):
    def build_graph_with_sub_graph(self):
        def linear_fc(num):
            data = fluid.layers.data(
                name='image', shape=[1, 32, 32], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            hidden = data
            for _ in six.moves.xrange(num):
                hidden = fluid.layers.fc(hidden, size=128, act='relu')
            loss = fluid.layers.cross_entropy(input=hidden, label=label)
            loss = fluid.layers.mean(loss)
            return loss

        main_program = Program()
        startup_program = Program()

        def true_func():
            return linear_fc(3)

        def false_func():
            return linear_fc(5)

        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = layers.less_than(y, x)
            out = layers.cond(pred, true_func, false_func)

        core_graph = core.Graph(main_program.desc)
        # We should create graph for test, otherwise it will throw a 
        # error that it cannot find the node of "STEP_COUNTER"
        graph = IrGraph(core_graph, for_test=True)
        sub_graph = graph.get_sub_graph(0)
        all_sub_graphs = graph.all_sub_graphs(
            for_test=True)  # same reason for subgraph
        # Should return graph and sub_graphs at the same time. If only return sub_graph, the graph will
        # be destructed and the sub_graphs will be empty.
        return graph, all_sub_graphs

    def test_quant_sub_graphs(self):
        graph, sub_graphs = self.build_graph_with_sub_graph()
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            place=fluid.CUDAPlace(0),
            activation_quantize_type='abs_max',
            weight_quantize_type='range_abs_max')
        Find_inserted_quant_op = False
        for sub_graph in sub_graphs:
            transform_pass.apply(sub_graph)
            for op in sub_graph.all_op_nodes():
                if 'quantize' in op.name():
                    Find_inserted_quant_op = True
        self.assertTrue(Find_inserted_quant_op)


if __name__ == '__main__':
    unittest.main()

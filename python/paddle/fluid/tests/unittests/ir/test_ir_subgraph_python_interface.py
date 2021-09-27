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

        def cond(i, ten):
            return i < ten

        def body(i, ten):
            i = i + 1
            return [i, ten]

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64')  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64')  # loop length
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
            # loss = linear_fc(3)
            # opt = fluid.optimizer.Adam(learning_rate=0.001)
            # opt.minimize(loss)
        core_graph = core.Graph(main_program.desc)
        print("block_size = ", len(main_program.blocks))
        print(core_graph)
        graph = IrGraph(core_graph)
        sub_graph = graph.get_sub_graph(0)
        all_sub_graphs = graph.all_sub_graphs()
        print(graph)
        print("all_sub_graphs[0]", all_sub_graphs[0])
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            place=fluid.CUDAPlace(0),
            activation_quantize_type='abs_max',
            weight_quantize_type='range_abs_max')
        print("sub_graph_size:", len(all_sub_graphs))
        for sub_graph in all_sub_graphs:
            print("sub_graph_dict:", sub_graph.__dict__)  # subgraph_
            op_nodes = sub_graph.all_op_nodes()
            var_nodes = sub_graph.all_var_nodes()
            print("inside func:ir graph's core graph", sub_graph.graph)
            print("all nodes inside build graph func:",
                  op_nodes)  #这里 all nodes 不为空，所以可以作量化Pass
            print("all var_nodes inside build graph func:", var_nodes)
            transform_pass.apply(sub_graph)

            print("Done quant pass applied ")
        return all_sub_graphs
        #return [graph]
    def test_quant_sub_graphs(self):
        sub_graphs = self.build_graph_with_sub_graph()
        print("num of sub graphs:", len(sub_graphs))
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            place=fluid.CUDAPlace(0),
            activation_quantize_type='abs_max',
            weight_quantize_type='range_abs_max')
        for sub_graph in sub_graphs:
            print("sub_graph_dict_outside:", sub_graph.
                  __dict__)  # 这里object地址都和返回前一样，但是拿到的all_op_nodes为空set()
            nodes = sub_graph.all_nodes()
            print("ir graph's core graph", sub_graph.graph)
            print("all nodes outside build graph func:", nodes)
            transform_pass.apply(sub_graph)  #因为拿不到 all_op_nodes 所以pass会失败
            #program = sub_graph.to_program()
        # print("sub_program",program)


if __name__ == '__main__':
    unittest.main()

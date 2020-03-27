#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass


class TestQat2Int8MkldnnPass(unittest.TestCase):
    def setUp(self):
        self.scope = fluid.Scope()
        self.place = fluid.CPUPlace()
        self.dtype = np.float32
        self.use_cudnn = False
        self.use_mkldnn = True
        self.data_format = "ANYLAYOUT"
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.groups = 1
        self.input_size = [1, 3, 5, 5]
        self.filter_size = [16, 3, 3, 3]
        self.filter_size2 = [1, 16, 2, 2]
        self.conv_output_size = [1, 16, 3, 3]
        self.conv_output2_size = [1, 1, 2, 2]
        self.input = np.random.random(self.input_size).astype(self.dtype)
        self.filter = np.random.random(self.filter_size).astype(self.dtype)
        self.filter2 = np.random.random(self.filter_size2).astype(self.dtype)
        self.conv_output = np.ndarray(self.conv_output_size).astype(self.dtype)
        self.conv_output2 = np.ndarray(self.conv_output2_size).astype(
            self.dtype)
        self.quantized_ops = 'conv2d'
        self.variables = {
            "input": self.input,
            "filter": self.filter,
            "filter2": self.filter2,
            "conv_output": self.conv_output,
            "conv_output2": self.conv_output2,
        }

    def prepare_program(self, program):
        block = program.global_block()
        for name in self.variables:
            block.create_var(
                name=name, dtype="float32", shape=self.variables[name].shape)
        conv2d_op1 = block.append_op(
            type="conv2d",
            inputs={
                "Input": block.var('input'),
                'Filter': block.var('filter')
            },
            outputs={"Output": block.var('conv_output')},
            attrs={
                'strides': self.stride,
                'paddings': self.pad,
                'groups': self.groups,
                'dilations': self.dilations,
                'use_cudnn': self.use_cudnn,
                'use_mkldnn': self.use_mkldnn,
                'data_format': self.data_format,
                'fuse_relu': True
            })
        conv2d_op2 = block.append_op(
            type="conv2d",
            inputs={
                "Input": block.var('conv_output'),
                'Filter': block.var('filter2')
            },
            outputs={"Output": block.var('conv_output2')},
            attrs={
                'strides': self.stride,
                'paddings': self.pad,
                'groups': self.groups,
                'dilations': self.dilations,
                'use_cudnn': self.use_cudnn,
                'use_mkldnn': self.use_mkldnn,
                'data_format': self.data_format,
                'fuse_brelu': True
            })

    def remove_fuse_activation_attribute(self, graph):
        for op in graph.all_op_nodes():
            op.op().remove_attr("fuse_activation")
        return graph

    def check_graph_before_pass(self, graph):
        for op in graph.all_op_nodes():
            self.assertFalse(op.op().has_attr("fuse_activation"))

    def check_graph_after_pass(self, graph):
        for op in graph.all_op_nodes():
            self.assertTrue(op.op().has_attr("fuse_activation"))
            if op.op().has_attr("fuse_relu") and op.op().attr("fuse_relu"):
                self.assertTrue(op.op().attr("fuse_activation") == "relu")
            if op.op().has_attr("fuse_brelu") and op.op().attr("fuse_brelu"):
                self.assertTrue(op.op().attr("fuse_activation") == "relu6")

    def test_qat_update_activation(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            self.prepare_program(program)
            graph = IrGraph(core.Graph(program.desc), for_test=True)
            graph = self.remove_fuse_activation_attribute(graph)
            self.check_graph_before_pass(graph)
            qat2_int8_mkldnn_pass = Qat2Int8MkldnnPass(
                self.quantized_ops,
                _scope=self.scope,
                _place=self.place,
                _core=core,
                _debug=False)
            graph = qat2_int8_mkldnn_pass._update_activations(graph)
            self.check_graph_after_pass(graph)


if __name__ == '__main__':
    unittest.main()

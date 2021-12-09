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
from paddle.fluid.contrib.slim.quantization import Quant2Int8MkldnnPass
import paddle

paddle.enable_static()


class TestQuant2Int8MkldnnPassMul(unittest.TestCase):
    def op_name(self):
        return "mul"

    def setUp(self):
        self.scope = fluid.Scope()
        self.place = fluid.CPUPlace()
        self.dtype = np.float32
        self.use_mkldnn = True

        self.quantized_ops = self.op_name()
        self.mul_input_size = [1, 3]
        self.mul_weights_size = [3, 5]
        self.mul_output_size = [1, 5]
        self.mul_input = np.random.random(self.mul_input_size).astype(
            self.dtype)
        self.mul_weights = np.ones(self.mul_weights_size, self.dtype)
        self.mul_weights_bad = np.ones([1, 1], self.dtype)
        self.mul_output = np.ndarray(self.mul_output_size).astype(self.dtype)
        self.mul_output_scale = np.linspace(1, 5, num=5).astype(self.dtype)

        self.variables_mul = {
            "mul_input": self.mul_input,
            "mul_weights": self.mul_weights,
            "mul_output": self.mul_output,
            "mul_weights_bad": self.mul_weights_bad
        }

    def prepare_program_mul(self, program):
        block = program.global_block()
        for name in self.variables_mul:
            block.create_var(
                name=name,
                dtype="float32",
                shape=self.variables_mul[name].shape)

        mul_op1 = block.append_op(
            type=self.op_name(),
            inputs={
                "X": block.var('mul_input'),
                "Y": block.var('mul_weights')
            },
            outputs={"Out": block.var('mul_output')},
            attrs={'use_mkldnn': self.use_mkldnn})

    def test_dequantize_op_weights(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            self.prepare_program_mul(program)
            graph = IrGraph(core.Graph(program.desc), for_test=True)

            op_node = ""
            for op in graph.all_op_nodes():
                if op.op().type() == self.op_name():
                    op_node = op
                    break
            assert op_node != "", "op of type %s not found" % self.op_name()

            qpass = Quant2Int8MkldnnPass(
                self.quantized_ops,
                _scope=self.scope,
                _place=self.place,
                _core=core,
                _debug=False)
            qpass._weight_thresholds["mul_output"] = self.mul_output_scale
            param = self.scope.var("mul_weights").get_tensor()
            param.set(self.variables_mul["mul_weights"], self.place)
            qpass._dequantize_op_weights(graph, op_node, "Y", "Out")

            assert np.allclose(
                self.scope.find_var("mul_weights").get_tensor(),
                [[1. / 127., 2. / 127., 3. / 127., 4. / 127., 5. / 127.],
                 [1. / 127., 2. / 127., 3. / 127., 4. / 127., 5. / 127.],
                 [1. / 127., 2. / 127., 3. / 127., 4. / 127., 5. / 127.]])

            param = self.scope.var("mul_weights").get_tensor()
            param.set(self.variables_mul["mul_weights_bad"], self.place)
            with self.assertRaises(ValueError):
                qpass._dequantize_op_weights(graph, op_node, "Y", "Out")


class TestQuant2Int8MkldnnPassMatmulV2(TestQuant2Int8MkldnnPassMul):
    def op_name(self):
        return "matmul_v2"


class TestQuant2Int8MkldnnPassConv2D(unittest.TestCase):
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

    def prepare_program_conv2d(self, program):
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
            if op.op().type() == "conv2d":
                self.assertTrue(op.op().has_attr("fuse_activation"))
                if op.op().has_attr("fuse_relu") and op.op().attr("fuse_relu"):
                    self.assertTrue(op.op().attr("fuse_activation") == "relu")
                if op.op().has_attr("fuse_brelu") and op.op().attr(
                        "fuse_brelu"):
                    self.assertTrue(op.op().attr("fuse_activation") == "relu6")

    def test_quant_update_activation(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            self.prepare_program_conv2d(program)
            graph = IrGraph(core.Graph(program.desc), for_test=True)
            graph = self.remove_fuse_activation_attribute(graph)
            self.check_graph_before_pass(graph)
            quant2_int8_mkldnn_pass = Quant2Int8MkldnnPass(
                self.quantized_ops,
                _scope=self.scope,
                _place=self.place,
                _core=core,
                _debug=False)
            graph = quant2_int8_mkldnn_pass._update_activations(graph)
            self.check_graph_after_pass(graph)


if __name__ == '__main__':
    unittest.main()

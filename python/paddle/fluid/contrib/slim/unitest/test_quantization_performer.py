#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
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
import random
import numpy as np
import paddle
import paddle.fluid as fluid
import six
from paddle.fluid.framework import Program
from paddle.fluid.contrib.slim.quantization import QuantizationPerformer
from paddle.fluid.contrib.slim.graph import PyGraph
from paddle.fluid import core


def linear_fc(num):
    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in six.moves.xrange(num):
        hidden = fluid.layers.fc(hidden, size=128, act='relu')
    loss = fluid.layers.cross_entropy(input=hidden, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def residual_block(num):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in six.moves.xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10)
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestQuantizationPerformer(unittest.TestCase):
    def setUp(self):
        # since quant_op and dequant_op is not ready, use cos and sin for test
        self.weight_quant_op_type = 'fake_quantize_abs_max'
        self.dequant_op_type = 'fake_dequantize_max_abs'
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y']
        }
        self.quantizable_op_grad_and_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y']
        }

    def linear_fc_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        graph = PyGraph(core.Graph(main.desc))
        performer = QuantizationPerformer(activation_quantize_type=quant_type)
        performer.quantize_transform(graph, False)
        marked_nodes = set()
        for op in graph.all_ops():
            if op.name().find('quantize') > -1:
                marked_nodes.add(op)
        graph.draw_graph('.', 'quantize_fc_' + quant_type, marked_nodes)

    def test_linear_fc_quant_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.linear_fc_quant('abs_max')

    def test_linear_fc_quant_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.linear_fc_quant('range_abs_max')

    def residual_block_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        graph = PyGraph(core.Graph(main.desc))
        performer = QuantizationPerformer(activation_quantize_type=quant_type)
        performer.quantize_transform(graph, False)
        marked_nodes = set()
        for op in graph.all_ops():
            if op.name().find('quantize') > -1:
                marked_nodes.add(op)
        graph.draw_graph('.', 'quantize_residual_' + quant_type, marked_nodes)

    def test_residual_block_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.residual_block_quant('abs_max')

    def test_residual_block_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.residual_block_quant('range_abs_max')


if __name__ == '__main__':
    unittest.main()

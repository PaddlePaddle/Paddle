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

import os
import unittest
import random
import numpy as np
import paddle.fluid as fluid
import six
import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import TransformForMkldnnPass
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid import core

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


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


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss


class TestTransformForMkldnnPass(unittest.TestCase):
    def setUp(self):
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            #Temporarily not support mul int8 kernel
            #'mul': ['X', 'Y']
        }

    def check_program(self, mkldnn_int8_pass, program):
        quantized_ops = set()
        for block in program.blocks:
            for op in block.ops:
                if op.type in self.quantizable_op_and_inputs:
                    for arg_name in op.output_arg_names:
                        self.assertTrue(arg_name.endswith('.dequantized'))
                        quantized_ops.add(arg_name)

    def residual_block_quant(self, activation_quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        graph = IrGraph(core.Graph(main.desc), for_test=True)
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type)
        transform_pass.apply(graph)
        mkldnn_int8_pass = TransformForMkldnnPass(
            scope=fluid.global_scope(), place=place)
        mkldnn_int8_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            graph.draw('.', 'quantize_residual_' + activation_quant_type,
                       marked_nodes)

        program = graph.to_program()
        self.check_program(program)

    def test_residual_block_range_abs_max(self):
        self.residual_block_quant('range_abs_max')

    def test_residual_block_moving_average_abs_max(self):
        self.residual_block_quant('moving_average_abs_max')


if __name__ == '__main__':
    unittest.main()

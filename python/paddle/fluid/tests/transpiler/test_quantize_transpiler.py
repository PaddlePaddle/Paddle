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

import numpy as np
import unittest
import paddle.fluid as fluid


def linear_fc(num):
    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in xrange(num):
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
    for _ in xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10)
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestQuantizeTranspiler(unittest.TestCase):
    def setUp(self):
        # since quant_op and dequant_op is not ready, use cos and sin for test
        self.quant_op_type = 'fake_quantize'
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

    def check_program(self, program):
        quantized_ops = {}
        for block in program.blocks:
            for idx, op in enumerate(block.ops):
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for i, arg_name in enumerate(op.input_arg_names):
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        if arg_name not in quantized_ops:
                            self.assertEqual(block.ops[idx - 2 * i - 1].type,
                                             self.dequant_op_type)
                            self.assertEqual(block.ops[idx - 2 * i - 2].type,
                                             self.quant_op_type)
                            quantized_ops[arg_name] = block.ops[idx - 2 * i - 2]
                        else:
                            op_idx = block.ops.index(quantized_ops[arg_name])
                            self.assertLess(op_idx, idx)

                # check backward
                if op.type in self.quantizable_op_grad_and_inputs:
                    for pname in self.quantizable_op_grad_and_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        self.assertTrue(arg_name in quantized_ops)

    def test_linear_fc(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            #f = open("raw_program", 'w+')
            #print >> f, main
            t = fluid.QuantizeTranspiler()
            t.transpile(main)
            #f = open("quanted_program", 'w+')
            #print >> f, main
            self.check_program(main)

    def test_residual_block(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            f = open("raw_program", 'w+')
            print >> f, main
            t = fluid.QuantizeTranspiler()
            t.transpile(main)
            f = open("quanted_program", 'w+')
            print >> f, main
            self.check_program(main)


if __name__ == '__main__':
    unittest.main()

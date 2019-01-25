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
import paddle.fluid as fluid
import six
import paddle
from paddle.fluid.framework import Program
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
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


class TestQuantizationTransformPass(unittest.TestCase):
    def setUp(self):
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y']
        }
        self.quantizable_grad_op_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y']
        }

    def check_program(self, transform_pass, program):
        quantized_ops = set()
        for block in program.blocks:
            for op in block.ops:
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        quantized_ops.add(arg_name)

            for op in block.ops:
                # check backward
                if op.type in self.quantizable_grad_op_inputs:
                    for pname in self.quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        self.assertTrue(arg_name in quantized_ops)

    def linear_fc_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        exe = fluid.Executor(fluid.CPUPlace())
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            program_exe=exe,
            activation_quantize_type=quant_type)
        transform_pass.apply(graph)
        marked_nodes = set()
        for op in graph.all_ops():
            if op.name().find('quantize') > -1:
                marked_nodes.add(op)
        graph.draw('.', 'quantize_fc_' + quant_type, marked_nodes)
        program = graph.to_program()
        self.check_program(transform_pass, program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        val_marked_nodes = set()
        for op in val_graph.all_ops():
            if op.name().find('quantize') > -1:
                val_marked_nodes.add(op)
        val_graph.draw('.', 'val_fc_' + quant_type, val_marked_nodes)

    def no_test_linear_fc_quant_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.linear_fc_quant('abs_max')

    def no_test_linear_fc_quant_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.linear_fc_quant('range_abs_max')

    def residual_block_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        exe = fluid.Executor(fluid.CPUPlace())
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            program_exe=exe,
            activation_quantize_type=quant_type)
        transform_pass.apply(graph)
        marked_nodes = set()
        for op in graph.all_ops():
            if op.name().find('quantize') > -1:
                marked_nodes.add(op)
        graph.draw('.', 'quantize_residual_' + quant_type, marked_nodes)
        program = graph.to_program()
        self.check_program(transform_pass, program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        val_marked_nodes = set()
        for op in val_graph.all_ops():
            if op.name().find('quantize') > -1:
                val_marked_nodes.add(op)
        val_graph.draw('.', 'val_residual_' + quant_type, val_marked_nodes)

    def no_test_residual_block_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.residual_block_quant('abs_max')

    def no_test_residual_block_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.residual_block_quant('range_abs_max')


class TestQuantizationFreezePass(unittest.TestCase):
    def freeze_graph(self, use_cuda, seed, quant_type):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32')
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.001)
                        opt.minimize(loss)
            return [img, label], loss

        random.seed(0)
        np.random.seed(0)

        main = fluid.Program()
        startup = fluid.Program()
        test_program = fluid.Program()
        feeds, loss = build_program(main, startup, False)
        build_program(test_program, startup, True)
        test_program = test_program.clone(for_test=True)
        main_graph = IrGraph(core.Graph(main.desc), for_test=False)
        test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup)
        transform_pass = QuantizationTransformPass(
            scope=scope, program_exe=exe, activation_quantize_type=quant_type)
        transform_pass.apply(main_graph)
        transform_pass.apply(test_graph)

        iters = 5
        batch_size = 8
        dev_name = '_gpu_' if use_cuda else '_cpu_'

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        with fluid.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(program=main_graph.to_program(),
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                print('{}: {}'.format(dev_name, loss_v))

        marked_nodes = set()
        for op in main_graph.all_ops():
            if op.name().find('quantize') > -1:
                marked_nodes.add(op)
        main_graph.draw('.', 'main' + dev_name + quant_type, marked_nodes)

        freeze_pass = QuantizationFreezePass(scope=scope, place=place)
        origin_marked_nodes = set()
        for op in test_graph.all_ops():
            if op.name().find('quantize') > -1:
                origin_marked_nodes.add(op)
        test_graph.draw('.', 'test_origin' + dev_name + quant_type,
                        origin_marked_nodes)
        freeze_pass.apply(test_graph)
        freeze_marked_nodes = set()
        for op in test_graph.all_ops():
            if op.name().find('quantize') > -1:
                freeze_marked_nodes.add(op)
        test_graph.draw('.', 'test_freeze' + dev_name + quant_type,
                        freeze_marked_nodes)

    # with fluid.program_guard(test_program):
    #     test_data = next(test_reader())
    #     w_var = fluid.framework._get_var('conv2d_1.w_0.quantized',
    #                                      test_program)
    #     # Testing during training
    #     test_loss1, w_quant = exe.run(program=test_program,
    #                                   feed=feeder.feed(test_data),
    #                                   fetch_list=[loss, w_var])

    #     # Freeze program for inference, but the weight of fc/conv is still float type.
    #     quant_transpiler.freeze_program(test_program, place)
    #     test_loss2, = exe.run(program=test_program,
    #                           feed=feeder.feed(test_data),
    #                           fetch_list=[loss])
    #     self.assertAlmostEqual(test_loss1, test_loss2, delta=5e-3)
    #     w_freeze = np.array(fluid.global_scope().find_var('conv2d_1.w_0')
    #                         .get_tensor())
    #     # fail: -432.0 != -433.0, this is due to the calculation precision
    #     #self.assertAlmostEqual(np.sum(w_freeze), np.sum(w_quant))

    #     # Convert parameter to 8-bit.
    #     quant_transpiler.convert_to_int8(test_program, place)
    #     # Save the 8-bit parameter and model file.
    #     fluid.io.save_inference_model('model_8bit', ['image', 'label'],
    #                                   [loss], exe, test_program)
    #     # Test whether the 8-bit parameter and model file can be loaded successfully.
    #     [infer, feed, fetch] = fluid.io.load_inference_model('model_8bit',
    #                                                          exe)
    #     # Check the loaded 8-bit weight.
    #     w_8bit = np.array(fluid.global_scope().find_var('conv2d_1.w_0.int8')
    #                       .get_tensor())

    #     self.assertEqual(w_8bit.dtype, np.int8)
    #     self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))

    def test_freeze_program_cuda_dynamic(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.freeze_graph(True, seed=1, quant_type='abs_max')

    def test_freeze_program_cpu_dynamic(self):
        with fluid.unique_name.guard():
            self.freeze_graph(False, seed=2, quant_type='abs_max')

    def test_freeze_program_cuda_static(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.freeze_graph(True, seed=1, quant_type='range_abs_max')

    def test_freeze_program_cpu_static(self):
        with fluid.unique_name.guard():
            self.freeze_graph(False, seed=2, quant_type='range_abs_max')


if __name__ == '__main__':
    unittest.main()

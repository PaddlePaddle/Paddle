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
from paddle.fluid.contrib.slim.quantization.quantization_performer import QuantizationPerformer
from paddle.fluid.contrib.slim.quantization.quantization_performer import _original_var_name
from paddle.fluid.contrib.slim.graph.executor import get_executor
from paddle.fluid.contrib.slim.graph import ImitationGraph
from paddle.fluid.contrib.slim.graph import load_inference_graph_model
from paddle.fluid.contrib.slim.graph import save_inference_graph_model


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

    def check_graph(self, graph):
        quantized_ops = {}

        persistable_vars = [
            v.name
            for v in filter(lambda var: var.persistable, graph.all_vars())
        ]

        for idx, op in enumerate(graph.all_ops()):
            # check forward
            if op.type in self.quantizable_op_and_inputs:
                for i, arg_name in enumerate(op.input_arg_names):
                    quant_op_type = self.weight_quant_op_type if \
                        _original_var_name(arg_name) \
                        in persistable_vars else self.act_quant_op_type
                    self.assertTrue(arg_name.endswith('.quantized.dequantized'))
                    if arg_name not in quantized_ops:
                        self.assertEqual(graph.all_ops()[idx - 2 * i - 1].type,
                                         self.dequant_op_type)
                        self.assertEqual(graph.all_ops()[idx - 2 * i - 2].type,
                                         quant_op_type)
                        quantized_ops[arg_name] = graph.all_ops()[idx - 2 * i -
                                                                  2]
                    else:
                        op_idx = graph.all_ops().index(quantized_ops[arg_name])
                        self.assertLess(op_idx, idx)

            # check backward
            if op.type in self.quantizable_op_grad_and_inputs:
                for pname in self.quantizable_op_grad_and_inputs[op.type]:
                    arg_name = op.input(pname)[0]
                    self.assertTrue(arg_name.endswith('.quantized.dequantized'))
                    self.assertTrue(arg_name in quantized_ops)

    def linear_fc_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            graph = ImitationGraph(main)
            performer = QuantizationPerformer(
                activation_quantize_type=quant_type)
            performer.quantize_transform(graph)
            self.check_graph(graph)

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
            graph = ImitationGraph(main)
            performer = QuantizationPerformer(
                activation_quantize_type=quant_type)
            performer.quantize_transform(graph)
            self.check_graph(graph)

    def test_residual_block_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.residual_block_quant('abs_max')

    def test_residual_block_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.residual_block_quant('range_abs_max')

    def freeze_graph(self, use_cuda, seed):
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
        feeds, loss = build_program(main, startup, False)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        program_exe = fluid.Executor(place)
        program_exe.run(startup)

        graph = ImitationGraph(main)
        performer = QuantizationPerformer()
        # performer = QuantizationPerformer(activation_quantize_type='range_abs_max')
        need_inited = performer.quantize_transform(graph)
        init_program = Program()
        for var, initializer in need_inited.iteritems():
            init_program.global_block()._clone_variable(var)
            initializer(var, init_program.global_block())
        program_exe.run(program=init_program, scope=fluid.global_scope())

        iters = 5
        batch_size = 8
        class_num = 10

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        exe = get_executor(graph, place)

        for _ in range(iters):
            data = next(train_reader())
            _ = exe.run(graph=graph,
                        feed=feeder.feed(data),
                        fetches=[loss.name])

        test_graph = graph.clone(for_test=True).prune(feeds, loss)
        w_var = test_graph.var('conv2d_1.w_0.quantized')
        test_data = next(test_reader())
        # Testing during training
        test_loss1, w_quant = exe.run(graph=test_graph,
                                      feed=feeder.feed(test_data),
                                      fetches=[loss.name, w_var.name])

        # Freeze program for inference, but the weight of fc/conv is still float type.
        performer.freeze_graph(test_graph, place, fluid.global_scope())
        test_loss2, = exe.run(graph=test_graph,
                              feed=feeder.feed(test_data),
                              fetches=[loss.name])
        self.assertAlmostEqual(test_loss1, test_loss2, delta=5e-3)
        w_freeze = np.array(fluid.global_scope().find_var('conv2d_1.w_0')
                            .get_tensor())
        self.assertAlmostEqual(np.sum(w_freeze), np.sum(w_quant))

        # Convert parameter to 8-bit.
        # Save the 8-bit parameter and model file.
        performer.convert_to_int8(test_graph, place, fluid.global_scope())
        save_inference_graph_model('model_8bit', ['image', 'label'],
                                   [loss.name], exe, test_graph)
        # Test whether the 8-bit parameter and model file can be loaded successfully.
        [infer, feed, fetch] = load_inference_graph_model('model_8bit', exe)
        # Check the loaded 8-bit weight.
        w_8bit = np.array(fluid.global_scope().find_var('conv2d_1.w_0.int8')
                          .get_tensor())

        self.assertEqual(w_8bit.dtype, np.int8)
        self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))

    def test_freeze_graph_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.freeze_graph(True, seed=1)

    def test_freeze_graph_cpu(self):
        with fluid.unique_name.guard():
            self.freeze_graph(False, seed=2)


if __name__ == '__main__':
    unittest.main()

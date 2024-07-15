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
import random
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base.framework import IrGraph
from paddle.framework import core
from paddle.static.quantization import (
    AddQuantDequantPass,
    ConvertToInt8Pass,
    QuantizationFreezePass,
    QuantizationTransformPass,
    QuantizationTransformPassV2,
    TransformForMobilePass,
)

paddle.enable_static()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


def linear_fc(num):
    data = paddle.static.data(
        name='image', shape=[-1, 1, 32, 32], dtype='float32'
    )
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = data
    for _ in range(num):
        hidden = paddle.static.nn.fc(hidden, size=128, activation='relu')
    loss = paddle.nn.functional.cross_entropy(
        input=hidden, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def residual_block(num, quant_skip_pattern=None):
    def conv_bn_layer(
        input, ch_out, filter_size, stride, padding, act='relu', bias_attr=False
    ):
        tmp = paddle.static.nn.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr,
        )
        return paddle.static.nn.batch_norm(input=tmp, act=act)

    data = paddle.static.data(
        name='image',
        shape=[1, 1, 32, 32],
        dtype='float32',
    )
    label = paddle.static.data(name='label', shape=[1, 1], dtype='int64')
    hidden = data
    for _ in range(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = paddle.add(x=conv, y=short)
        hidden = paddle.nn.functional.relu(hidden)
    matmul_weight = paddle.static.create_parameter(
        shape=[1, 16, 32, 32], dtype='float32'
    )
    hidden = paddle.matmul(hidden, matmul_weight, True, True)
    if quant_skip_pattern:
        with paddle.static.name_scope(quant_skip_pattern):
            pool = paddle.nn.functional.avg_pool2d(
                hidden, kernel_size=2, stride=2
            )
    else:
        pool = paddle.nn.functional.avg_pool2d(hidden, kernel_size=2, stride=2)
    fc = paddle.static.nn.fc(pool, size=10)
    loss = paddle.nn.functional.cross_entropy(
        input=fc, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def conv_net(img, label, quant_skip_pattern):
    conv_out_1 = paddle.static.nn.conv2d(
        input=img,
        filter_size=5,
        num_filters=20,
        act='relu',
    )
    conv_pool_1 = paddle.nn.functional.max_pool2d(
        conv_out_1, kernel_size=2, stride=2
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)

    conv_out_2 = paddle.static.nn.conv2d(
        input=conv_pool_1,
        filter_size=5,
        num_filters=20,
        act='relu',
    )
    conv_pool_2 = paddle.nn.functional.avg_pool2d(
        conv_out_2, kernel_size=2, stride=2
    )
    hidden = paddle.static.nn.fc(conv_pool_2, size=100, activation='relu')
    with paddle.static.name_scope(quant_skip_pattern):
        prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


class TestQuantizationTransformPass(unittest.TestCase):
    def setUp(self):
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y'],
        }
        self.quantizable_grad_op_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y'],
        }

    def check_program(self, program):
        quantized_ops = set()
        for block in program.blocks:
            for op in block.ops:
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        quantized_ops.add(arg_name)

            for op in block.ops:
                # check backward
                if op.type in self.quantizable_grad_op_inputs:
                    for pname in self.quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        self.assertTrue(arg_name in quantized_ops)

    def linear_fc_quant(
        self, activation_quant_type, weight_quantize_type, for_ci=True
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            loss = linear_fc(3)
            opt = paddle.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        place = paddle.CPUPlace()
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=paddle.static.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quantize_type,
        )
        transform_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            graph.draw(
                '.', 'quantize_fc_' + activation_quant_type, marked_nodes
            )
        program = graph.to_program()
        self.check_program(program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        if not for_ci:
            val_marked_nodes = set()
            for op in val_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    val_marked_nodes.add(op)
            val_graph.draw(
                '.', 'val_fc_' + activation_quant_type, val_marked_nodes
            )

    def test_linear_fc_quant_abs_max(self):
        self.linear_fc_quant('abs_max', 'abs_max', for_ci=True)

    def test_linear_fc_quant_range_abs_max(self):
        self.linear_fc_quant('range_abs_max', 'abs_max', for_ci=True)

    def test_linear_fc_quant_moving_average_abs_max(self):
        self.linear_fc_quant(
            'moving_average_abs_max', 'channel_wise_abs_max', for_ci=True
        )

    def residual_block_quant(
        self,
        activation_quant_type,
        weight_quantize_type,
        quantizable_op_type,
        for_ci=True,
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            loss = residual_block(2)
            opt = paddle.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        place = paddle.CPUPlace()
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=paddle.static.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quantize_type,
            quantizable_op_type=quantizable_op_type,
        )
        transform_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            graph.draw(
                '.', 'quantize_residual_' + activation_quant_type, marked_nodes
            )
        program = graph.to_program()
        self.check_program(program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        if not for_ci:
            val_marked_nodes = set()
            for op in val_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    val_marked_nodes.add(op)
            val_graph.draw(
                '.', 'val_residual_' + activation_quant_type, val_marked_nodes
            )

    def test_residual_block_abs_max(self):
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul']
        self.residual_block_quant(
            'abs_max', 'abs_max', quantizable_op_type, for_ci=True
        )

    def test_residual_block_range_abs_max(self):
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul']
        self.residual_block_quant(
            'range_abs_max', 'abs_max', quantizable_op_type, for_ci=True
        )

    def test_residual_block_moving_average_abs_max(self):
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul']
        self.residual_block_quant(
            'moving_average_abs_max',
            'channel_wise_abs_max',
            quantizable_op_type,
            for_ci=True,
        )


class TestQuantizationFreezePass(unittest.TestCase):
    def freeze_graph(
        self,
        use_cuda,
        seed,
        activation_quant_type,
        bias_correction=False,
        weight_quant_type='abs_max',
        for_ci=True,
        quant_skip_pattern='skip_quant',
    ):
        def build_program(main, startup, is_test):
            paddle.seed(seed)
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main, startup):
                    img = paddle.static.data(
                        name='image', shape=[-1, 1, 28, 28], dtype='float32'
                    )
                    label = paddle.static.data(
                        name='label', shape=[-1, 1], dtype='int64'
                    )
                    loss = conv_net(img, label, quant_skip_pattern)
                    if not is_test:
                        opt = paddle.optimizer.Adam(learning_rate=0.001)
                        opt.minimize(loss)
            return [img, label], loss

        random.seed(0)
        np.random.seed(0)

        main = paddle.static.Program()
        startup = paddle.static.Program()
        test_program = paddle.static.Program()
        feeds, loss = build_program(main, startup, False)
        build_program(test_program, startup, True)
        test_program = test_program.clone(for_test=True)
        main_graph = IrGraph(core.Graph(main.desc), for_test=False)
        test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)

        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.global_scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
            skip_pattern=quant_skip_pattern,
        )
        transform_pass.apply(main_graph)
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
            skip_pattern=quant_skip_pattern,
        )
        transform_pass.apply(test_graph)
        dev_name = '_gpu_' if use_cuda else '_cpu_'
        if not for_ci:
            marked_nodes = set()
            for op in main_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            main_graph.draw(
                '.',
                'main'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw(
                '.',
                'test'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        build_strategy.fuse_all_reduce_ops = False
        binary = paddle.static.CompiledProgram(
            main_graph.graph, build_strategy=build_strategy
        )
        quantized_test_program = test_graph.to_program()
        iters = 5
        batch_size = 8

        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size,
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        feeder = paddle.base.DataFeeder(feed_list=feeds, place=place)
        with paddle.static.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss]
                )
                if not for_ci:
                    print(
                        '{}: {}'.format(
                            'loss'
                            + dev_name
                            + activation_quant_type
                            + '_'
                            + weight_quant_type,
                            loss_v,
                        )
                    )

        test_data = next(test_reader())
        with paddle.static.program_guard(quantized_test_program):
            w_var = base.framework._get_var(
                'conv2d_1.w_0.quantized', quantized_test_program
            )
        # Testing
        with paddle.static.scope_guard(scope):
            test_loss1, w_quant = exe.run(
                program=quantized_test_program,
                feed=feeder.feed(test_data),
                fetch_list=[loss, w_var],
            )

        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope,
            place=place,
            bias_correction=bias_correction,
            weight_quantize_type=weight_quant_type,
        )
        freeze_pass.apply(test_graph)
        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw(
                '.',
                'test_freeze'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )

        server_program = test_graph.to_program()
        with paddle.static.scope_guard(scope):
            (test_loss2,) = exe.run(
                program=server_program,
                feed=feeder.feed(test_data),
                fetch_list=[loss],
            )
        self.assertAlmostEqual(test_loss1, test_loss2, delta=5e-3)
        if not for_ci:
            print(
                '{}: {}'.format(
                    'test_loss1'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    test_loss1,
                )
            )
            print(
                '{}: {}'.format(
                    'test_loss2'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    test_loss2,
                )
            )
        w_freeze = np.array(scope.find_var('conv2d_1.w_0').get_tensor())
        # Maybe failed, this is due to the calculation precision
        # self.assertAlmostEqual(np.sum(w_freeze), np.sum(w_quant))
        if not for_ci:
            print(
                '{}: {}'.format(
                    'w_freeze'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    np.sum(w_freeze),
                )
            )
            print(
                '{}: {}'.format(
                    'w_quant'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    np.sum(w_quant),
                )
            )

        # Convert parameter to 8-bit.
        convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
        convert_int8_pass.apply(test_graph)
        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw(
                '.',
                'test_int8'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )
        server_program_int8 = test_graph.to_program()
        # Save the 8-bit parameter and model file.
        with paddle.static.scope_guard(scope):
            feed_list = ['image', 'label']
            feed_vars = [
                server_program_int8.global_block().var(name)
                for name in feed_list
            ]
            paddle.static.save_inference_model(
                'server_int8'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type
                + '/model',
                feed_vars,
                [loss],
                exe,
                program=server_program_int8,
            )
            # Test whether the 8-bit parameter and model file can be loaded successfully.
            [infer, feed, fetch] = paddle.static.load_inference_model(
                'server_int8'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type
                + '/model',
                exe,
            )
        # Check the loaded 8-bit weight.
        w_8bit = np.array(scope.find_var('conv2d_1.w_0.int8').get_tensor())
        self.assertEqual(w_8bit.dtype, np.int8)
        self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))
        if not for_ci:
            print(
                '{}: {}'.format(
                    'w_8bit'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    np.sum(w_8bit),
                )
            )
            print(
                '{}: {}'.format(
                    'w_freeze'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    np.sum(w_freeze),
                )
            )

        mobile_pass = TransformForMobilePass()
        mobile_pass.apply(test_graph)
        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw(
                '.',
                'test_mobile'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )

        mobile_program = test_graph.to_program()
        with paddle.static.scope_guard(scope):
            feed_list = ['image', 'label']
            feed_vars = [
                mobile_program.global_block().var(name) for name in feed_list
            ]
            paddle.static.save_inference_model(
                'mobile_int8'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type
                + '/model',
                feed_vars,
                [loss],
                exe,
                program=mobile_program,
            )

    def test_freeze_graph_cuda_dynamic(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='abs_max',
                    weight_quant_type='abs_max',
                    for_ci=True,
                )
            with paddle.utils.unique_name.guard():
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                )

    def test_freeze_graph_cpu_dynamic(self):
        with paddle.utils.unique_name.guard():
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
            )

    def test_freeze_graph_cuda_static(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='range_abs_max',
                    bias_correction=True,
                    weight_quant_type='abs_max',
                    for_ci=True,
                )
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='range_abs_max',
                    weight_quant_type='abs_max',
                    for_ci=True,
                )
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='abs_max',
                    for_ci=True,
                )
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='range_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                )
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                )
                self.freeze_graph(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    bias_correction=True,
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                )

    def test_freeze_graph_cpu_static(self):
        with paddle.utils.unique_name.guard():
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='range_abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='range_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
            )
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
            )


def quant_dequant_residual_block(num, quant_skip_pattern=None):
    def conv_bn_layer(
        input, ch_out, filter_size, stride, padding, act='relu', bias_attr=False
    ):
        tmp = paddle.static.nn.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr,
        )
        return paddle.static.nn.batch_norm(input=tmp, act=act)

    data1 = paddle.static.data(
        name='image', shape=[-1, 1, 32, 32], dtype='float32'
    )
    data2 = paddle.static.data(
        name='matmul_input', shape=[-1, 16, 32, 32], dtype='float32'
    )
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = data1
    for _ in range(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = paddle.add(x=conv, y=short)
        hidden = paddle.nn.functional.relu(hidden)
    hidden = paddle.matmul(hidden, data2, True, True)
    if isinstance(quant_skip_pattern, str):
        with paddle.static.name_scope(quant_skip_pattern):
            pool1 = paddle.nn.functional.avg_pool2d(
                hidden, kernel_size=2, stride=2
            )
            pool2 = paddle.nn.functional.max_pool2d(
                hidden, kernel_size=2, stride=2
            )
            pool_add = paddle.add(pool1, pool2)
            pool_add = paddle.nn.functional.relu(pool_add)
    elif isinstance(quant_skip_pattern, list):
        assert (
            len(quant_skip_pattern) > 1
        ), 'test config error: the len of quant_skip_pattern list should be greater than 1.'
        with paddle.static.name_scope(quant_skip_pattern[0]):
            pool1 = paddle.nn.functional.avg_pool2d(
                hidden, kernel_size=2, stride=2
            )
            pool2 = paddle.nn.functional.max_pool2d(
                hidden, kernel_size=2, stride=2
            )
        with paddle.static.name_scope(quant_skip_pattern[1]):
            pool_add = paddle.add(pool1, pool2)
            pool_add = paddle.nn.functional.relu(pool_add)
    else:
        pool1 = paddle.nn.functional.avg_pool2d(hidden, kernel_size=2, stride=2)
        pool2 = paddle.nn.functional.max_pool2d(hidden, kernel_size=2, stride=2)
        pool_add = paddle.add(pool1, pool2)
        pool_add = paddle.nn.functional.relu(pool_add)
    fc = paddle.static.nn.fc(pool_add, size=10)
    loss = paddle.nn.functional.cross_entropy(
        input=fc, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


class TestAddQuantDequantPass(unittest.TestCase):
    def setUp(self):
        self._target_ops = {'elementwise_add', 'pool2d'}
        self._target_grad_ops = {'elementwise_add_grad', 'pool2d_grad'}

    def check_graph(self, graph, skip_pattern=None):
        ops = graph.all_op_nodes()
        for op_node in ops:
            if op_node.name() in self._target_ops:
                user_skipped = False
                if isinstance(skip_pattern, list):
                    user_skipped = op_node.op().has_attr(
                        "op_namescope"
                    ) and any(
                        pattern in op_node.op().attr("op_namescope")
                        for pattern in skip_pattern
                    )
                elif isinstance(skip_pattern, str):
                    user_skipped = (
                        op_node.op().has_attr("op_namescope")
                        and op_node.op().attr("op_namescope").find(skip_pattern)
                        != -1
                    )

                if user_skipped:
                    continue

                in_nodes_all_not_persistable = True
                for input_name in op_node.input_arg_names():
                    in_node = graph._find_node_by_name(
                        op_node.inputs, input_name
                    )
                    in_nodes_all_not_persistable = (
                        in_nodes_all_not_persistable
                        and not in_node.persistable()
                    )
                if not in_nodes_all_not_persistable:
                    continue
                input_names = op_node.input_arg_names()
                for input_name in input_names:
                    self.assertTrue(input_name.endswith('.quant_dequant'))

    def residual_block_quant(
        self, quantizable_op_type, skip_pattern=None, for_ci=True
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            loss = quant_dequant_residual_block(2, skip_pattern)
            opt = paddle.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        place = paddle.CPUPlace()
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        add_quant_dequant_pass = AddQuantDequantPass(
            scope=paddle.static.global_scope(),
            place=place,
            skip_pattern=skip_pattern,
            quantizable_op_type=quantizable_op_type,
        )
        add_quant_dequant_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quant') > -1:
                    marked_nodes.add(op)
            graph.draw('.', 'add_quant_dequant_graph', marked_nodes)
        self.check_graph(graph, skip_pattern)
        program = graph.to_program()
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        if not for_ci:
            val_marked_nodes = set()
            for op in val_graph.all_op_nodes():
                if op.name().find('quant') > -1:
                    val_marked_nodes.add(op)
            val_graph.draw('.', 'val_add_quant_dequant_graph', val_marked_nodes)

    def test_residual_block(self):
        quantizable_op_type = ['elementwise_add', 'pool2d', 'mul', 'matmul']
        self.residual_block_quant(
            quantizable_op_type, skip_pattern=None, for_ci=True
        )

    def test_residual_block_skip_pattern(self):
        quantizable_op_type = ['elementwise_add', 'pool2d', 'mul', 'matmul']
        self.residual_block_quant(
            quantizable_op_type, skip_pattern='skip_quant', for_ci=True
        )

    def test_residual_block_skip_pattern_1(self):
        quantizable_op_type = ['elementwise_add', 'pool2d', 'mul', 'matmul']
        self.residual_block_quant(
            quantizable_op_type,
            skip_pattern=['skip_quant1', 'skip_quant2'],
            for_ci=True,
        )


class TestQuantizationTransformPassV2(unittest.TestCase):
    def setUp(self):
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y'],
        }
        self.quantizable_grad_op_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y'],
        }

    def check_program(self, program):
        quantized_ops = set()
        for block in program.blocks:
            for op in block.ops:
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        quantized_ops.add(arg_name)

            for op in block.ops:
                # check backward
                if op.type in self.quantizable_grad_op_inputs:
                    for pname in self.quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        self.assertTrue(arg_name in quantized_ops)

    def linear_fc_quant(
        self, activation_quant_type, weight_quantize_type, for_ci=True
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            loss = linear_fc(3)
            opt = paddle.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        place = paddle.CPUPlace()
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPassV2(
            scope=paddle.static.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quantize_type,
        )
        transform_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            graph.draw(
                '.', 'quantize_fc_' + activation_quant_type, marked_nodes
            )
        program = graph.to_program()
        self.check_program(program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        if not for_ci:
            val_marked_nodes = set()
            for op in val_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    val_marked_nodes.add(op)
            val_graph.draw(
                '.', 'val_fc_' + activation_quant_type, val_marked_nodes
            )

    def test_linear_fc_quant_abs_max(self):
        self.linear_fc_quant('abs_max', 'abs_max', for_ci=True)

    def test_linear_fc_quant_channel_wise_abs_max(self):
        self.linear_fc_quant('abs_max', 'channel_wise_abs_max', for_ci=True)

    def residual_block_quant(
        self,
        activation_quant_type,
        weight_quantize_type,
        quantizable_op_type,
        for_ci=True,
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            loss = residual_block(2)
            opt = paddle.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        place = paddle.CPUPlace()
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=paddle.static.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quantize_type,
            quantizable_op_type=quantizable_op_type,
        )
        transform_pass.apply(graph)
        if not for_ci:
            marked_nodes = set()
            for op in graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            graph.draw(
                '.', 'quantize_residual_' + activation_quant_type, marked_nodes
            )
        program = graph.to_program()
        self.check_program(program)
        val_graph = IrGraph(core.Graph(program.desc), for_test=False)
        if not for_ci:
            val_marked_nodes = set()
            for op in val_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    val_marked_nodes.add(op)
            val_graph.draw(
                '.', 'val_residual_' + activation_quant_type, val_marked_nodes
            )

    def test_residual_block_abs_max(self):
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul']
        self.residual_block_quant(
            'abs_max', 'abs_max', quantizable_op_type, for_ci=True
        )

    def test_residual_block_channel_wise_abs_max(self):
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul']
        self.residual_block_quant(
            'abs_max', 'channel_wise_abs_max', quantizable_op_type, for_ci=True
        )


if __name__ == '__main__':
    unittest.main()

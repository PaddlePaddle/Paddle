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

import os
import random
import unittest

import numpy as np

import paddle
from paddle.base.framework import IrGraph
from paddle.framework import core
from paddle.static.quantization import (
    QuantInt8MkldnnPass,
    QuantizationFreezePass,
    QuantizationTransformPass,
)

paddle.enable_static()
os.environ["CPU_NUM"] = "1"


def conv_net(img, label):
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
    conv_pool_2 = paddle.nn.functional.max_pool2d(
        conv_out_2, kernel_size=2, stride=2
    )
    prediction = paddle.static.nn.fc(conv_pool_2, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


class TestMKLDNNTransformBasedFreezePass(unittest.TestCase):
    def setUp(self):
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y'],
        }

    def check_program(self, program):
        for block in program.blocks:
            for op in block.ops:
                if op.type in self.quantizable_op_and_inputs:
                    for arg_name in op.output_arg_names:
                        # Check quantizable op's output is linked to
                        # fake_dequantize's output
                        self.assertTrue(arg_name.endswith('.dequantized'))

    def isinteger(self, x):
        return np.equal(np.mod(x, 1), 0)

    def build_program(self, main, startup, is_test, seed):
        paddle.seed(seed)
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main, startup):
                img = paddle.static.data(
                    name='image', shape=[-1, 1, 28, 28], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                )
                loss = conv_net(img, label)
                if not is_test:
                    opt = paddle.optimizer.Adam(learning_rate=0.001)
                    opt.minimize(loss)
        return [img, label], loss

    def mkldnn_based_freeze_graph(
        self,
        use_cuda,
        seed,
        activation_quant_type,
        weight_quant_type='abs_max',
        quant_perf=False,
        for_ci=False,
    ):
        random.seed(0)
        np.random.seed(0)

        main = paddle.static.Program()
        startup = paddle.static.Program()
        test_program = paddle.static.Program()
        feeds, loss = self.build_program(main, startup, False, seed)
        self.build_program(test_program, startup, True, seed)
        test_program = test_program.clone(for_test=True)
        main_graph = IrGraph(core.Graph(main.desc), for_test=False)
        test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.global_scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
        # Apply the QuantizationTransformPass
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
        )
        transform_pass.apply(main_graph)
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
        )
        transform_pass.apply(test_graph)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
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

        # Training the model to get the weights value
        with paddle.static.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss]
                )

        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope, place=place, weight_quantize_type=weight_quant_type
        )
        freeze_pass.apply(test_graph)

        # Transform quantized graph for MKL-DNN INT8 inference
        mkldnn_int8_pass = QuantInt8MkldnnPass(_scope=scope, _place=place)
        mkldnn_int8_pass.apply(test_graph)
        dev_name = '_cpu_'
        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw(
                '.',
                'test_mkldnn'
                + dev_name
                + activation_quant_type
                + '_'
                + weight_quant_type,
                marked_nodes,
            )
        mkldnn_program = test_graph.to_program()

        # Check the transformation weights of conv2d and mul
        conv_w_mkldnn = np.array(scope.find_var('conv2d_1.w_0').get_tensor())
        mul_w_mkldnn = np.array(scope.find_var('fc_0.w_0').get_tensor())
        # Check if weights are still integer
        self.assertFalse(self.isinteger(np.sum(conv_w_mkldnn)))
        self.assertFalse(self.isinteger(np.sum(mul_w_mkldnn)))

        # Check if the conv2d output and mul output are correctly linked to fake_dequantize's
        # output
        self.check_program(mkldnn_program)
        if not for_ci:
            print(
                '{}: {}'.format(
                    'w_mkldnn'
                    + dev_name
                    + activation_quant_type
                    + '_'
                    + weight_quant_type,
                    np.sum(mul_w_mkldnn),
                )
            )

    def test_mkldnn_graph_cpu_static(self):
        with paddle.utils.unique_name.guard():
            self.mkldnn_based_freeze_graph(
                False,
                seed=2,
                activation_quant_type='range_abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )
            self.mkldnn_based_freeze_graph(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )


if __name__ == '__main__':
    unittest.main()

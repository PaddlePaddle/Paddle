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
import six
import paddle.fluid as fluid
import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ScaleForTrainingPass
from paddle.fluid.contrib.slim.quantization import ScaleForInferencePass
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass
from paddle.fluid import core

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


def residual_block(img, label, num=1):
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

    hidden = img
    for _ in six.moves.xrange(num):
        conv = conv_bn_layer(hidden, 20, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 20, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestQuantizationScalePass(unittest.TestCase):
    def quantization_scale(self,
                           use_cuda,
                           seed,
                           activation_quant_type,
                           weight_quant_type='abs_max',
                           for_ci=False):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32')
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    loss = residual_block(img, label, 1)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.0001)
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
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type)
        transform_pass.apply(main_graph)
        transform_pass.apply(test_graph)

        add_quant_dequant_pass = AddQuantDequantPass(scope=scope, place=place)
        add_quant_dequant_pass.apply(main_graph)
        add_quant_dequant_pass.apply(test_graph)

        scale_training_pass = ScaleForTrainingPass(scope=scope, place=place)
        scale_training_pass.apply(main_graph)

        dev_name = '_gpu' if use_cuda else '_cpu'
        if not for_ci:
            marked_nodes = set()
            for op in main_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            main_graph.draw('.', 'main_scale' + dev_name, marked_nodes)
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw('.', 'test_scale' + dev_name, marked_nodes)

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        binary = fluid.CompiledProgram(main_graph.graph).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)
        iters = 5
        batch_size = 8

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        with fluid.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(binary,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])
                if not for_ci:
                    print('{}: {}'.format('loss' + dev_name, loss_v))

        scale_inference_pass = ScaleForInferencePass(scope=scope)
        scale_inference_pass.apply(test_graph)

        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope, place=place, weight_quantize_type=weight_quant_type)
        freeze_pass.apply(test_graph)
        server_program = test_graph.to_program()

        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw('.', 'quant_scale' + dev_name, marked_nodes)

        with open('quant_scale_model' + dev_name + '.txt', 'w') as f:
            f.write(str(server_program))

        with fluid.scope_guard(scope):
            fluid.io.save_inference_model('quant_scale_model' + dev_name,
                                          ['image', 'label'], [loss], exe,
                                          server_program)

    def test_quant_scale_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True)

    def test_quant_scale_cpu(self):
        with fluid.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True)


if __name__ == '__main__':
    unittest.main()

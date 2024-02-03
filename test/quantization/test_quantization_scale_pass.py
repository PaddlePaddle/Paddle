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
import tempfile
import unittest

import numpy as np

import paddle
from paddle.base.framework import IrGraph
from paddle.framework import core
from paddle.static.quantization import (
    AddQuantDequantPass,
    OutScaleForInferencePass,
    OutScaleForTrainingPass,
    QuantizationFreezePass,
    QuantizationTransformPass,
)

paddle.enable_static()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    conv_pool_2 = paddle.nn.functional.avg_pool2d(
        conv_out_2, kernel_size=2, stride=2
    )
    hidden = paddle.static.nn.fc(conv_pool_2, size=100, activation='relu')
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


class TestQuantizationScalePass(unittest.TestCase):
    def quantization_scale(
        self,
        use_cuda,
        seed,
        activation_quant_type,
        weight_quant_type='abs_max',
        for_ci=False,
    ):
        def build_program(main, startup, is_test):
            paddle.seed(2023)
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
                        opt = paddle.optimizer.Adam(learning_rate=0.0001)
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
        )
        transform_pass.apply(main_graph)
        transform_pass.apply(test_graph)

        add_quant_dequant_pass = AddQuantDequantPass(scope=scope, place=place)
        add_quant_dequant_pass.apply(main_graph)
        add_quant_dequant_pass.apply(test_graph)

        scale_training_pass = OutScaleForTrainingPass(scope=scope, place=place)
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

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        build_strategy.fuse_all_reduce_ops = False
        binary = paddle.static.CompiledProgram(
            main_graph.graph, build_strategy=build_strategy
        )
        iters = 5
        batch_size = 8

        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size,
        )
        feeder = paddle.base.DataFeeder(feed_list=feeds, place=place)
        with paddle.static.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss]
                )
                if not for_ci:
                    print('{}: {}'.format('loss' + dev_name, loss_v))

        scale_inference_pass = OutScaleForInferencePass(scope=scope)
        scale_inference_pass.apply(test_graph)

        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope, place=place, weight_quantize_type=weight_quant_type
        )
        freeze_pass.apply(test_graph)
        server_program = test_graph.to_program()

        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw('.', 'quant_scale' + dev_name, marked_nodes)

        tempdir = tempfile.TemporaryDirectory()
        mapping_table_path = os.path.join(
            tempdir.name, 'quant_scale_model' + dev_name + '.txt'
        )
        save_path = os.path.join(tempdir.name, 'quant_scale_model' + dev_name)
        with open(mapping_table_path, 'w') as f:
            f.write(str(server_program))

        with paddle.static.scope_guard(scope):
            feed_list = ['image', 'label']
            feed_vars = [
                server_program.global_block().var(name) for name in feed_list
            ]
            paddle.static.save_inference_model(
                save_path,
                feed_vars,
                [loss],
                exe,
                program=server_program,
                clip_extra=True,
            )
        tempdir.cleanup()

    def test_quant_scale_cuda(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                )

    def test_quant_scale_cpu(self):
        with paddle.utils.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
            )


if __name__ == '__main__':
    unittest.main()

#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
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

import json
import os
import random
import tempfile
import unittest

import numpy as np

import paddle
from paddle.base.framework import IrGraph
from paddle.framework import LayerHelper, core
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


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = paddle.ParamAttr(
        name=x.name + '_pact',
        initializer=paddle.nn.initializer.Constant(value=init_thres),
        regularizer=paddle.regularizer.L2Decay(0.0001),
        learning_rate=1,
    )
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = paddle.subtract(
        x, paddle.nn.functional.relu(paddle.subtract(x, u_param))
    )
    x = paddle.add(x, paddle.nn.functional.relu(paddle.subtract(-u_param, x)))

    return x


class TestUserDefinedQuantization(unittest.TestCase):
    def quantization_scale(
        self,
        use_cuda,
        seed,
        activation_quant_type,
        weight_quant_type='abs_max',
        for_ci=False,
        act_preprocess_func=None,
        weight_preprocess_func=None,
        act_quantize_func=None,
        weight_quantize_func=None,
    ):
        def build_program(main, startup, is_test):
            paddle.seed(seed)
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main, startup):
                    img = paddle.static.data(
                        name='image', shape=[-1, 1, 28, 28], dtype='float32'
                    )
                    img.stop_gradient = False
                    label = paddle.static.data(
                        name='label', shape=[-1, 1], dtype='int64'
                    )
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = paddle.optimizer.SGD(learning_rate=0.0001)
                        opt.minimize(loss)
            return [img, label], loss

        def get_optimizer():
            return paddle.optimizer.Momentum(0.0001, 0.9)

        def load_dict(mapping_table_path):
            with open(mapping_table_path, 'r') as file:
                data = file.read()
                data = json.loads(data)
                return data

        def save_dict(Dict, mapping_table_path):
            with open(mapping_table_path, 'w') as file:
                file.write(json.dumps(Dict))

        random.seed(0)
        np.random.seed(0)
        tempdir = tempfile.TemporaryDirectory()
        mapping_table_path = os.path.join(tempdir.name, 'inference')

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
        train_transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
            act_preprocess_func=act_preprocess_func,
            weight_preprocess_func=weight_preprocess_func,
            act_quantize_func=act_quantize_func,
            weight_quantize_func=weight_quantize_func,
            optimizer_func=get_optimizer,
            executor=exe,
        )
        train_transform_pass.apply(main_graph)
        test_transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
            act_preprocess_func=act_preprocess_func,
            weight_preprocess_func=weight_preprocess_func,
            act_quantize_func=act_quantize_func,
            weight_quantize_func=weight_quantize_func,
            optimizer_func=get_optimizer,
            executor=exe,
        )

        test_transform_pass.apply(test_graph)
        save_dict(test_graph.out_node_mapping_table, mapping_table_path)

        add_quant_dequant_pass = AddQuantDequantPass(scope=scope, place=place)
        add_quant_dequant_pass.apply(main_graph)
        add_quant_dequant_pass.apply(test_graph)

        scale_training_pass = OutScaleForTrainingPass(scope=scope, place=place)
        scale_training_pass.apply(main_graph)

        dev_name = '_gpu' if use_cuda else '_cpu'

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

        out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
        out_scale_infer_pass.apply(test_graph)

        freeze_pass = QuantizationFreezePass(
            scope=scope,
            place=place,
            weight_bits=8,
            activation_bits=8,
            weight_quantize_type=weight_quant_type,
        )

        mapping_table = load_dict(mapping_table_path)
        test_graph.out_node_mapping_table = mapping_table
        if act_quantize_func is None and weight_quantize_func is None:
            freeze_pass.apply(test_graph)
        tempdir.cleanup()

    def test_act_preprocess_cuda(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                    act_preprocess_func=pact,
                )

    def test_act_preprocess_cpu(self):
        with paddle.utils.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
                act_preprocess_func=pact,
            )

    def test_weight_preprocess_cuda(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                    weight_preprocess_func=pact,
                )

    def test_weight_preprocess_cpu(self):
        with paddle.utils.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
                weight_preprocess_func=pact,
            )

    def test_act_quantize_cuda(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                    act_quantize_func=pact,
                )

    def test_act_quantize_cpu(self):
        with paddle.utils.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
                act_quantize_func=pact,
            )

    def test_weight_quantize_cuda(self):
        if core.is_compiled_with_cuda():
            with paddle.utils.unique_name.guard():
                self.quantization_scale(
                    True,
                    seed=1,
                    activation_quant_type='moving_average_abs_max',
                    weight_quant_type='channel_wise_abs_max',
                    for_ci=True,
                    weight_quantize_func=pact,
                )

    def test_weight_quantize_cpu(self):
        with paddle.utils.unique_name.guard():
            self.quantization_scale(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
                weight_quantize_func=pact,
            )


if __name__ == '__main__':
    unittest.main()
